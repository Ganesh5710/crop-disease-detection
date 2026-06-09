import React from "react";

function ConfidenceBar({ confidence }) {
  return (
    <div className="confidence-container">

      <div
        className="confidence-fill"
        style={{
          width: `${confidence}%`
        }}
      >
        {confidence}%
      </div>

    </div>
  );
}

export default ConfidenceBar;