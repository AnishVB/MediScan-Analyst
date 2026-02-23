import { useRef, useState, useCallback } from "react";

function ImageUploader({
  selectedImage,
  imagePreview,
  onImageSelect,
  onRemoveImage,
  onAnalyze,
  isAnalyzing,
}) {
  const fileInputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("image/")) {
        onImageSelect(file);
      }
    },
    [onImageSelect],
  );

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleFileChange = useCallback(
    (e) => {
      const file = e.target.files[0];
      if (file) onImageSelect(file);
    },
    [onImageSelect],
  );

  return (
    <div className="uploader">
      <div className="uploader-title">Upload Medical Image</div>

      {!selectedImage ? (
        <div
          className={`upload-zone ${isDragging ? "dragging" : ""}`}
          onClick={() => fileInputRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          <div className="upload-icon">📤</div>
          <div className="upload-text">
            <span className="accent">Click to upload</span> or drag and drop
          </div>
          <div className="upload-hint">
            X-ray, MRI, CT Scan — PNG, JPG, WEBP
          </div>
        </div>
      ) : (
        <div className="image-preview">
          <img src={imagePreview} alt="Medical scan preview" />
          <div className="image-preview-overlay">
            <span className="image-name">{selectedImage.name}</span>
            <button className="btn-remove" onClick={onRemoveImage}>
              ✕ Remove
            </button>
          </div>
        </div>
      )}

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        style={{ display: "none" }}
      />

      <button
        className={`btn-analyze ${isAnalyzing ? "analyzing" : ""}`}
        onClick={onAnalyze}
        disabled={!selectedImage || isAnalyzing}
      >
        {isAnalyzing ? (
          <>
            <span className="spinner" />
            Agents Processing...
          </>
        ) : (
          <>🔬 Run Multi-Agent Analysis</>
        )}
      </button>
    </div>
  );
}

export default ImageUploader;
