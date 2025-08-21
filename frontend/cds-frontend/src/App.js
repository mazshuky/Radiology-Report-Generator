import React, { useState } from "react";
import "./App.css";

// Simple UI components (since you might not have shadcn/ui set up)
const Card = ({ children, className }) => (
  <div className={`border rounded-lg shadow-md ${className}`}>{children}</div>
);

const CardContent = ({ children, className }) => (
  <div className={`p-4 ${className}`}>{children}</div>
);

const Button = ({ children, disabled, onClick, type, className }) => (
  <button
    type={type}
    disabled={disabled}
    onClick={onClick}
    className={`px-4 py-2 bg-blue-500 text-white rounded disabled:opacity-50 ${className}`}
  >
    {children}
  </button>
);

const Input = ({ type, placeholder, value, onChange, className }) => (
  <input
    type={type}
    placeholder={placeholder}
    value={value}
    onChange={onChange}
    className={`w-full px-3 py-2 border rounded-md ${className}`}
  />
);

const Textarea = ({ placeholder, value, onChange, className }) => (
  <textarea
    placeholder={placeholder}
    value={value}
    onChange={onChange}
    className={`w-full px-3 py-2 border rounded-md ${className}`}
    rows={3}
  />
);

export default function CDSApp() {
  const [file, setFile] = useState(null);
  const [age, setAge] = useState(60);
  const [sex, setSex] = useState("Male");
  const [symptoms, setSymptoms] = useState("");
  const [mode, setMode] = useState("clinician");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("age", age);
    formData.append("sex", sex);
    formData.append("symptoms", symptoms);
    formData.append("mode", mode);

    try {
      const res = await fetch("http://localhost:8000/predict_and_report", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      alert("Error connecting to backend");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <Card className="max-w-3xl mx-auto">
        <CardContent className="space-y-4">
          <h1 className="text-2xl font-bold mb-4">Clinical Decision Support</h1>

          <form onSubmit={handleSubmit} className="space-y-3">
            <Input type="file" onChange={(e) => setFile(e.target.files[0])} />
            <Input
              type="number"
              placeholder="Age"
              value={age}
              onChange={(e) => setAge(e.target.value)}
            />
            <Input
              type="text"
              placeholder="Sex (Male/Female)"
              value={sex}
              onChange={(e) => setSex(e.target.value)}
            />
            <Textarea
              placeholder="Symptoms (comma-separated)"
              value={symptoms}
              onChange={(e) => setSymptoms(e.target.value)}
            />
            <select
              className="w-full px-3 py-2 border rounded-md"
              value={mode}
              onChange={(e) => setMode(e.target.value)}
            >
              <option value="clinician">Clinician Mode</option>
              <option value="patient">Patient Mode</option>
            </select>

            <Button type="submit" disabled={loading}>
              {loading ? "Analyzing..." : "Upload & Analyze"}
            </Button>
          </form>

          {result && (
            <div className="mt-6 space-y-4">
              <h2 className="text-xl font-semibold">AI-Generated Report</h2>
              <div className="bg-gray-100 p-3 rounded space-y-3">
                {/* Patient Info */}
                <div>
                  <span className="font-medium">Patient Info:</span>
                  <ul className="ml-4 list-disc">
                    <li>Age: {age}</li>
                    <li>Sex: {sex}</li>
                    <li>
                      Symptoms:{" "}
                      {symptoms
                        ? symptoms
                            .split(",")
                            .map((s) => s.trim())
                            .filter(Boolean)
                            .join(", ")
                        : <span className="italic text-gray-500">None</span>}
                    </li>
                  </ul>
                </div>

                {/* Model Probabilities */}
                <div>
                  <span className="font-medium">Model Probabilities (Top 5):</span>
                  <ul className="ml-4 list-disc">
                    {Object.entries(result.prediction.probabilities)
                      .sort((a, b) => b[1] - a[1])
                      .slice(0, 5)
                      .map(([cls, prob]) => (
                        <li key={cls}>
                          {cls}: {prob.toFixed(3)}
                        </li>
                      ))}
                  </ul>
                </div>

                {/* Grad-CAM Overlays */}
                <div>
                  <span className="font-medium">Grad-CAM Overlays:</span>
                  <ul className="ml-4 list-disc">
                    {Object.entries(result.prediction.overlays).map(([cls, url]) => (
                      <li key={cls}>
                        <a
                          href={`http://localhost:8000${url}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-600 underline"
                        >
                          {cls} overlay
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Impression */}
                <div className="mt-3 p-3 bg-white border rounded shadow">
                  <span className="font-medium text-lg block mb-2">Impression</span>
                  <div className="prose max-w-none whitespace-pre-line mt-1">
                    {result.report.report}
                  </div>
                  <p className="text-xs text-gray-600 mt-2">{result.report.disclaimer}</p>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}