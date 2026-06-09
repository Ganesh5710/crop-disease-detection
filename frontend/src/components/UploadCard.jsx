import React from "react";

function UploadCard({
  preview,
  handleFileChange,
  predictDisease,
  loading
}) {

  return (
    <div className="upload-card">

      <input
        type="file"
        accept="image/*"
        onChange={handleFileChange}
      />

      {preview && (
        <img
          src={preview}
          alt="Preview"
          className="preview"
        />
      )}

      <br />

      <button
        onClick={predictDisease}
      >
        Detect Disease
      </button>

      {loading && (
        <h3>
          🔍 Analyzing Image...
        </h3>
      )}

    </div>
  );
}

export default UploadCard;