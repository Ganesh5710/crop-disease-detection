import React from "react";
import diseaseInfo from "../data/diseaseInfo";

function DiseaseInfo({ disease }) {

  const info = diseaseInfo[disease];

  if (!info) {
    return (
      <div className="disease-info">
        <p>
          No additional information available.
        </p>
      </div>
    );
  }

  return (
    <div className="disease-info">

      <h3>Symptoms</h3>
      <p>{info.symptoms}</p>

      <h3>Treatment</h3>
      <p>{info.treatment}</p>

      <h3>Prevention</h3>
      <p>{info.prevention}</p>

    </div>
  );
}

export default DiseaseInfo;