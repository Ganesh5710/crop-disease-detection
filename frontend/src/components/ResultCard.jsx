import React from "react";
import ConfidenceBar from "./ConfidenceBar";
import diseaseInfo from "../data/diseaseInfo";

function ResultCard({ result }) {

  const info =
    diseaseInfo[result.disease];

  return (
    <div className="result-card">

      <h2>
        Prediction Result
      </h2>

      <h3>
        🦠 {result.disease}
      </h3>

      <ConfidenceBar
        confidence={result.confidence}
      />

      {info && (
        <div className="disease-info">

          <h4>Symptoms</h4>
          <p>{info.symptoms}</p>

          <h4>Treatment</h4>
          <p>{info.treatment}</p>

          <h4>Prevention</h4>
          <p>{info.prevention}</p>

        </div>
      )}

    </div>
  );
}

export default ResultCard;