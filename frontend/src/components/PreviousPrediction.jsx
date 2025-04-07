import React, { useState, useEffect, useContext } from "react";
import { AuthContext } from "../context/AuthContext";
import Squares from "./Squares";

const PreviousPredictions = () => {

  const BASE_URL = import.meta.env.VITE_BASE_URL;
  
  const { user, isAuthenticated } = useContext(AuthContext);
  const [predictions, setPredictions] = useState([]);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const fetchPreviousPredictions = async () => {
    if (!isAuthenticated || !user || !user.id) {
      setError("User not authenticated or missing user ID");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${BASE_URL}/prediction/previous-predictions?user_id=${user.id}`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to fetch predictions");
      }

      const data = await response.json();
      setPredictions(data.predictions);
    } catch (err) {
      setError(err.message);
      console.error("Error fetching predictions:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const deletePrediction = async (id) => {
    if (!user || !user.id) {
      setError("User not authenticated");
      return;
    }
    
    if (!window.confirm("Are you sure you want to delete this prediction?")) return;

    try {
      const response = await fetch(`${BASE_URL}/prediction/delete_prediction`, {
        method: "DELETE",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ user_id: user.id, prediction_id: id }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to delete prediction");
      }

      setPredictions(predictions.filter((prediction) => prediction._id !== id));
    } catch (err) {
      console.error("Error deleting prediction:", err);
      setError(err.message);
    }
  };

  useEffect(() => {
    if (isAuthenticated && user && user.id) {
      fetchPreviousPredictions();
    }
  }, [isAuthenticated, user]);

  // Render confidence scores based on prediction type
  const renderConfidenceScores = (prediction) => {
    const scores = prediction.confidence_scores || {};
    const predictionType = prediction.prediction_type || "ml";
    
    if (predictionType === "rule-based") {
      return (
        <div>Rule-based: {scores.rule_based ? `${(scores.rule_based * 100).toFixed(2)}%` : "N/A"}</div>
      );
    } else {
      return (
        <>
          <div>RF: {scores.rf ? `${(scores.rf * 100).toFixed(2)}%` : "N/A"}</div>
          <div>NB: {scores.nb ? `${(scores.nb * 100).toFixed(2)}%` : "N/A"}</div>
          <div>SVM: {scores.svm ? `${(scores.svm * 100).toFixed(2)}%` : "N/A"}</div>
        </>
      );
    }
  };

  // Render final predictions based on prediction type
  const renderPredictions = (prediction) => {
    const finalPrediction = prediction.final_prediction;
    const predictionType = prediction.prediction_type || "ml";
    
    if (predictionType === "rule-based" || Array.isArray(finalPrediction)) {
      return (
        <div>
          {Array.isArray(finalPrediction) 
            ? finalPrediction.map((pred, idx) => (
                <div key={idx}>{pred}</div>
              ))
            : finalPrediction}
        </div>
      );
    } else {
      const fp = prediction.final_prediction || {};
      return (
        <>
          <div>RF: {fp.rf_prediction || "N/A"}</div>
          <div>NB: {fp.nb_prediction || "N/A"}</div>
          <div>SVM: {fp.svm_prediction || "N/A"}</div>
        </>
      );
    }
  };

  if (!isAuthenticated) return <div>Please log in to view your predictions.</div>;
  if (isLoading) return <div>Loading predictions...</div>;
  if (error) return <div className="text-red-500">{error}</div>;

  return (
    <div className="relative min-h-screen px-6 py-4">
      <div className="fixed inset-0 -z-10">
        <Squares speed={0.2} squareSize={40} direction="diagonal" borderColor="#3b82f6" hoverFillColor="#1e3a8a" />
      </div>

      <div className="relative w-full mx-auto bg-white/70 backdrop-blur-md border border-blue-300/50 shadow-lg rounded-lg overflow-hidden mt-4">
        <div className="px-6 py-4 bg-blue-100/50 border-b border-blue-300/50">
          <h2 className="text-lg font-semibold text-blue-600">Previous Predictions</h2>
        </div>

        <div className="px-4 py-4 overflow-y-auto">
          {predictions.length === 0 ? (
            <p className="text-blue-500">No previous predictions found.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full border-collapse bg-white/90 rounded-lg shadow-md">
                <thead>
                  <tr className="bg-blue-200/50 text-blue-800">
                    <th className="px-4 py-2 text-left text-sm font-semibold">Date</th>
                    <th className="px-4 py-2 text-left text-sm font-semibold">Symptoms</th>
                    <th className="px-4 py-2 text-left text-sm font-semibold">Disease</th>
                    <th className="px-4 py-2 text-left text-sm font-semibold">Prediction Type</th>
                    <th className="px-4 py-2 text-left text-sm font-semibold">Confidence</th>
                    <th className="px-4 py-2 text-left text-sm font-semibold">Predictions</th>
                    <th className="px-4 py-2 text-center text-sm font-semibold">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-blue-300/50">
                  {predictions.map((prediction) => (
                    <tr key={prediction._id} className="hover:bg-blue-100/50 transition-all">
                      <td className="px-4 py-2">{new Date(prediction.created_at).toLocaleDateString()}</td>
                      <td className="px-4 py-2">{prediction.symptoms.join(", ")}</td>
                      <td className="px-4 py-2 text-blue-700 font-medium">{prediction.disease_name}</td>
                      <td className="px-4 py-2">
                        <span className={`px-2 py-1 rounded text-xs font-medium ${prediction.prediction_type === "rule-based" ? "bg-green-100 text-green-800" : "bg-blue-100 text-blue-800"}`}>
                          {prediction.prediction_type === "rule-based" ? "Rule-based" : "ML"}
                        </span>
                      </td>
                      <td className="px-4 py-2">
                        {renderConfidenceScores(prediction)}
                      </td>
                      <td className="px-4 py-2">
                        {renderPredictions(prediction)}
                      </td>
                      <td className="px-4 py-2 text-center">
                        <button onClick={() => deletePrediction(prediction._id)} className="text-red-500 hover:text-red-700">
                          ❌
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PreviousPredictions;