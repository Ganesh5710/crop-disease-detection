import React, { useState } from "react";
import axios from "axios";

import "./App.css";

import Hero from "./components/Hero";
import UploadCard from "./components/UploadCard";
import ResultCard from "./components/ResultCard";

function App() {

  const [file, setFile] = useState(null);

  const [preview, setPreview] =
    useState("");

  const [result, setResult] =
    useState(null);

  const [loading, setLoading] =
    useState(false);

  const handleFileChange = (e) => {

    const selectedFile =
      e.target.files[0];

    if (!selectedFile) return;

    setFile(selectedFile);

    setPreview(
      URL.createObjectURL(
        selectedFile
      )
    );

    setResult(null);
  };

  const predictDisease = async () => {

    if (!file) {
      alert(
        "Please select an image"
      );
      return;
    }

    const formData =
      new FormData();

    formData.append(
      "file",
      file
    );

    try {

      setLoading(true);

      const response =
        await axios.post(
          "http://127.0.0.1:8000/predict",
          formData,
          {
            headers: {
              "Content-Type":
                "multipart/form-data"
            }
          }
        );

      setResult(
        response.data
      );

    } catch (error) {

      console.error(error);

      alert(
        "Prediction failed"
      );

    } finally {

      setLoading(false);
    }
  };

  return (
    <div className="container">

      <Hero />

      <UploadCard
        preview={preview}
        handleFileChange={
          handleFileChange
        }
        predictDisease={
          predictDisease
        }
        loading={loading}
      />

      {result && (
        <ResultCard
          result={result}
        />
      )}

    </div>
  );
}

export default App;