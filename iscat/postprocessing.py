import numpy as np
import cv2
from tqdm import tqdm


def compute_contrast_frames(signal_frames, reference_frames, params):
    """
    Compute per-frame contrast images prior to intensity windowing and
    normalization.

    This helper implements the background subtraction strategies described in
    the Code Design Document (CDD Section 3.5), but stops after producing
    floating-point contrast frames. It is a direct refactor of the background
    subtraction logic previously in apply_background_subtraction.

    Supported methods (selected via params["background_subtraction_method"]):

        - "reference_frame":
            For each frame, compute:
                Contrast = (Signal - Reference) / (Reference + eps)
            using the corresponding noisy reference frame.

        - "video_median":
            Estimate a static background frame from the entire video and
            subtract this background from each signal frame. The background is
            the per-pixel temporal median of all raw signal frames:
                B(x, y) = median_t Signal_t(x, y)
                Contrast_t(x, y) = Signal_t(x, y) - B(x, y)

          For backward compatibility with older configurations, the legacy
          option string "video_mean" is accepted as an exact alias for
          "video_median". The implementation always uses the temporal median
          (never the arithmetic mean).

    After the method-specific background subtraction, a final centering step
    is applied: for each contrast frame, its spatial median value is subtracted
    from all pixels. This removes residual per-frame DC offsets (caused by
    finite PSF support, numerical effects, or imperfect cancellation) that
    would otherwise appear as global brightness shifts from frame to frame,
    while preserving local contrast structure.

    Edge cases:
        - If signal_frames is empty, returns an empty list.
        - For "reference_frame", if reference_frames is empty, returns an empty
          list (no subtraction can be performed).
        - For unknown methods, raises ValueError.

    Args:
        signal_frames (list of np.ndarray): List of raw signal frames
            (typically uint16).
        reference_frames (list of np.ndarray): List of raw reference frames
            (typically uint16). Required for "reference_frame" method.
        params (dict): Global simulation parameter dictionary (PARAMS). Must
            contain "background_subtraction_method".

    Returns:
        list of np.ndarray: List of floating-point contrast frames. May be an
        empty list in the edge cases described above.
    """
    if not signal_frames:
        return []

    method = params.get("background_subtraction_method", "reference_frame")

    contrast_frames = []

    if method == "reference_frame":
        # This preserves the existing behavior:
        # Contrast = (Signal - Reference) / Reference
        if not reference_frames:
            # Cannot perform reference-frame subtraction without references.
            return []

        print("Applying background subtraction using per-frame reference images...")
        for signal_frame, ref_frame in tqdm(
            zip(signal_frames, reference_frames),
            total=len(signal_frames)
        ):
            signal_f = signal_frame.astype(float)
            ref_f = ref_frame.astype(float)
            # Add a small epsilon to the denominator to prevent division by zero.
            subtracted = (signal_f - ref_f) / (ref_f + 1e-9)
            contrast_frames.append(subtracted)

    elif method in ("video_median", "video_mean"):
        # Robust background subtraction from a single video:
        #   1. Build a per-pixel temporal median background frame:
        #        B(x,y) = median_t Signal_t(x,y)
        #   2. Subtract this background from each frame:
        #        Contrast_t(x,y) = Signal_t(x,y) - B(x,y)
        #
        # The reference frames are not used in this method.
        print(
            "Applying background subtraction using temporal median of signal "
            "frames (video_median method)..."
        )

        num_frames = len(signal_frames)
        frame_shape = signal_frames[0].shape

        # Stack all signal frames into a 3D array for per-pixel median calculation.
        # Use float32 to balance precision and memory usage (matches previous code).
        signal_stack = np.empty(
            (num_frames, frame_shape[0], frame_shape[1]), dtype=np.float32
        )
        for idx, frame in enumerate(signal_frames):
            # Automatic upcast from uint16 to float32.
            signal_stack[idx] = frame

        # Compute the per-pixel temporal median as the background estimate.
        background_frame = np.median(signal_stack, axis=0)

        # Subtract the background from each frame to obtain contrast frames.
        for frame in tqdm(signal_frames, total=num_frames):
            subtracted = frame.astype(np.float32) - background_frame
            contrast_frames.append(subtracted)

    else:
        raise ValueError(
            f"Unknown background_subtraction_method: {method}. "
            "Supported values are 'reference_frame' and 'video_median' "
            "(with 'video_mean' accepted as a deprecated alias)."
        )

    # --- Per-frame baseline removal to suppress global brightness swings ---
    # For each contrast frame, subtract its spatial median so that the
    # background baseline is centered around zero in every frame. This
    # suppresses small per-frame DC offsets that would otherwise appear as
    # uniform brightness changes across the whole image.
    for idx, frame in enumerate(contrast_frames):
        # Use the median (not the mean) to be robust against the small fraction
        # of pixels containing strong particle signal.
        median_val = np.median(frame)
        if median_val != 0.0:
            contrast_frames[idx] = frame - median_val

    return contrast_frames


def compute_single_frame_contrast(signal_frame, reference_frame, params):
    """
    Compute a single-frame contrast image from raw signal and reference
    frames, using the same background_subtraction_method semantics as
    compute_contrast_frames for the 'reference_frame' case, but without any
    temporal statistics or global percentile normalization.

    This helper is intended for interactive/viewer-style tools that operate on
    a single frame at a time. It focuses on producing a physically meaningful
    contrast map for that frame only.

    Behavior:
        - Inputs are converted to float32 for computation.
        - For background_subtraction_method == 'reference_frame':
              contrast = (signal - reference) / (reference + eps)
          where eps is a small constant to avoid division by zero.
        - For 'video_median' or 'video_mean', a single frame does not provide
          a temporal stack to form a median; these methods are not supported
          here and will raise a ValueError with a clear message.
        - For any other method string, a ValueError is raised.

        - No per-frame median subtraction or percentile windowing is applied
          here; the caller may choose how to visualize or further normalize
          the resulting contrast field.

    Args:
        signal_frame (np.ndarray): 2D array, raw signal frame (uint16/float).
        reference_frame (np.ndarray): 2D array, raw reference frame
            (uint16/float).
        params (dict): PARAMS-like dictionary providing
            'background_subtraction_method'.

    Returns:
        np.ndarray: 2D float32 contrast image.
    """
    if signal_frame is None or reference_frame is None:
        raise ValueError(
            "signal_frame and reference_frame must be provided for single-frame contrast."
        )

    signal_f = np.asarray(signal_frame, dtype=np.float32)
    reference_f = np.asarray(reference_frame, dtype=np.float32)

    method = params.get("background_subtraction_method", "reference_frame")

    if method == "reference_frame":
        contrast = (signal_f - reference_f) / (reference_f + 1e-9)
        return contrast.astype(np.float32)

    if method in ("video_median", "video_mean"):
        raise ValueError(
            "compute_single_frame_contrast does not support 'video_median' or "
            "'video_mean' for single-frame operation. Use 'reference_frame' "
            "for viewer-style tools."
        )

    raise ValueError(
        f"Unknown background_subtraction_method for single-frame contrast: {method}."
    )


def compute_single_frame_views(signal_frame, reference_frame, params):
    """
    Compute raw and contrast views for a single frame without global
    normalization or video encoding.

    Inputs:
        signal_frame   : 2D array-like, raw signal frame (uint16 or float).
        reference_frame: 2D array-like, raw reference frame (uint16 or float).
        params         : PARAMS-like dict, used only for
                         'background_subtraction_method'.

    Behavior:
        - raw_signal and raw_reference are float64 views of the inputs.
        - contrast is computed according to background_subtraction_method, but
          restricted to this single pair of frames and without temporal
          statistics:

            * "reference_frame":
                contrast = (signal - reference) / (reference + eps)

            * "video_median" (or "video_mean"):
                For a single-frame viewer, there is no temporal stack to form a
                median; we return a simple zero-mean version of the signal:

                    contrast = signal - median(signal)

          This keeps the function usable for any configured method while
          avoiding misleading semantics for single-frame "video_median".

        - A per-frame spatial median is subtracted from contrast, mirroring
          compute_contrast_frames, to remove residual DC offset.

    Returns:
        dict with keys:
            "raw_signal"    : 2D float64 array
            "raw_reference" : 2D float64 array
            "contrast"      : 2D float64 array
    """
    if signal_frame is None or reference_frame is None:
        raise ValueError("signal_frame and reference_frame must be provided for single-frame views.")

    signal_f = np.asarray(signal_frame, dtype=float)
    reference_f = np.asarray(reference_frame, dtype=float)

    method = params.get("background_subtraction_method", "reference_frame")

    if method == "reference_frame":
        contrast = (signal_f - reference_f) / (reference_f + 1e-9)
    elif method in ("video_median", "video_mean"):
        # Degenerate single-frame version of video_median: remove spatial
        # median so that the contrast is zero-centered but still reflects the
        # raw signal structure.
        median_val = np.median(signal_f)
        contrast = signal_f - median_val
    else:
        raise ValueError(
            f"Unknown background_subtraction_method for single-frame views: {method}."
        )

    # Per-frame baseline removal (robust against particle signal).
    frame_median = np.median(contrast)
    if frame_median != 0.0:
        contrast = contrast - frame_median

    return {
        "raw_signal": signal_f,
        "raw_reference": reference_f,
        "contrast": contrast,
    }


def compute_normalization_range(contrast_frames):
    """
    Compute the global lower and upper bounds used to map contrast frames to
    an 8-bit [0, 255] range.

    This function centralizes the normalization-range calculation so that
    alternative high-performance algorithms (CDD Section 6.3) can be
    introduced here without changing the rest of the post-processing
    pipeline.

    Current behavior:
        - Uses percentile-based windowing as described in CDD Section 3.5.
        - Computes the 0.5 and 99.5 percentile values over the entire set of
          contrast frames:
              min_val = P_0.5(contrast_frames)
              max_val = P_99.5(contrast_frames)

        - These values are returned as floats and are used unchanged by
          normalize_contrast_frames to map contrast frames into [0, 255].

    This implementation is mathematically identical to the previous inline
    use of np.percentile in normalize_contrast_frames; only the structure
    has been refactored.

    Args:
        contrast_frames (list of np.ndarray): List of floating-point contrast
            frames. Must be non-empty.

    Returns:
        tuple[float, float]: (min_val, max_val) percentile values.

    Raises:
        ValueError: If contrast_frames is empty.
    """
    if not contrast_frames:
        raise ValueError(
            "contrast_frames must be non-empty when computing a normalization range."
        )

    # Normalize the contrast range across the whole video for consistent brightness.
    # This robustly clips outliers by finding the 0.5 and 99.5 percentile values.
    min_val, max_val = np.percentile(contrast_frames, [0.5, 99.5])

    return float(min_val), float(max_val)


def normalize_contrast_frames(contrast_frames, original_frame_shape):
    """
    Normalize contrast frames to an 8-bit [0, 255] range using global
    percentile-based windowing.

    This helper implements the intensity windowing and normalization behavior
    described in CDD Section 3.5 and used previously at the end of
    apply_background_subtraction:

        1. Determine the 0.5 and 99.5 percentile values across the entire set
           of contrast frames. This robustly defines the minimum and maximum
           interesting signal.
        2. Normalize each contrast frame to [0, 255] using these values,
           clipping out-of-range values.
        3. In the degenerate case where max_val <= min_val, return a stack of
           constant mid-gray frames (value 128) with the same shape as the
           original images.

    The range-determination step is now delegated to compute_normalization_range,
    which preserves behavior but decouples the choice of algorithm from the
    normalization pipeline.

    Args:
        contrast_frames (list of np.ndarray): List of floating-point contrast
            frames.
        original_frame_shape (tuple[int, int]): Shape (height, width) of the
            original frames, used to construct fallback frames in the
            degenerate case.

    Returns:
        list of np.ndarray: List of uint8 frames normalized to [0, 255].
    """
    if not contrast_frames:
        return []

    # Compute the global normalization bounds using the centralized helper.
    min_val, max_val = compute_normalization_range(contrast_frames)

    final_frames_8bit = []
    if max_val > min_val:
        for frame in contrast_frames:
            # Scale the frame data to the 0-255 range.
            norm_frame = 255 * (frame - min_val) / (max_val - min_val)
            final_frames_8bit.append(
                np.clip(norm_frame, 0, 255).astype(np.uint8)
            )
    else:
        # Handle the edge case of a video with no contrast.
        final_frames_8bit = [
            np.full(original_frame_shape, 128, dtype=np.uint8)
            for _ in contrast_frames
        ]

    return final_frames_8bit


def apply_background_subtraction(signal_frames, reference_frames, params):
    """
    Perform background subtraction and normalize the result to an 8-bit range
    for video encoding.

    This function is the public entry point for post-processing and maintains
    the original behavior, but internally it is now factored into two steps:

        1. compute_contrast_frames(...):
               Computes floating-point contrast frames according to the selected
               background subtraction method ("reference_frame" or
               "video_median"; the legacy string "video_mean" is accepted as an
               alias for "video_median" and uses the temporal median).
        2. normalize_contrast_frames(...):
               Applies percentile-based intensity windowing (0.5 and 99.5
               percentiles) and maps the contrast frames into 8-bit [0, 255].

    Behavior (unchanged from the previous implementation, apart from the
    per-frame DC centering applied inside compute_contrast_frames):

        - If signal_frames is empty, returns an empty list.
        - For method "reference_frame" with missing reference_frames, returns
          an empty list.
        - For unknown methods, raises ValueError.
        - Otherwise, returns a list of 8-bit frames ready for video encoding.

    Args:
        signal_frames (list of np.ndarray): List of raw signal frames
            (typically uint16).
        reference_frames (list of np.ndarray): List of raw reference frames
            (typically uint16). Required for 'reference_frame' method.
        params (dict): Global simulation parameter dictionary (PARAMS). Must
            contain "background_subtraction_method".

    Returns:
        list of np.ndarray: List of 8-bit, normalized frames ready for video
            encoding. Returns an empty list if the inputs are empty or if
            reference frames are missing for the 'reference_frame' method.
    """
    if not signal_frames:
        return []

    # Step 1: compute floating-point contrast frames via the selected method.
    contrast_frames = compute_contrast_frames(signal_frames, reference_frames, params)
    if not contrast_frames:
        # This covers both the empty-input case (already handled above) and the
        # "reference_frame" case with missing reference frames.
        return []

    # Step 2: normalize contrast frames to 8-bit for video encoding.
    original_shape = signal_frames[0].shape
    final_frames_8bit = normalize_contrast_frames(contrast_frames, original_shape)

    return final_frames_8bit


def save_video(filename, frames, fps, size):
    """
    Encodes and saves a list of frames to an .mp4 video file.

    Args:
        filename (str): The output path for the video file.
        frames (list of np.ndarray): A list of 8-bit numpy array frames.
        fps (int or float): The desired frames per second for the video.
        size (tuple): The (width, height) of the video.
    """
    print(f"Saving final video to {filename}...")
    # Define the codec and create VideoWriter object.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(filename, fourcc, fps, size, isColor=False)

    for frame in frames:
        video_writer.write(frame)

    video_writer.release()
    print("Simulation finished successfully!")