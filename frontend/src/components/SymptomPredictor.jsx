import { useState, useEffect, useContext } from 'react';
import { X } from 'lucide-react';
import { LifeLine } from 'react-loading-indicators';
import { AuthContext } from '../context/AuthContext';

const SymptomPredictor = ({ onPredictionResult }) => {

  const BASE_URL = import.meta.env.VITE_BASE_URL;

  const [inputValue, setInputValue] = useState('');
  const [symptoms, setSymptoms] = useState([]);
  const [suggestions, setSuggestions] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [allSymptoms, setAllSymptoms] = useState([]);
  const [fileError, setFileError] = useState(null);

  const { user, isAuthenticated } = useContext(AuthContext);

  useEffect(() => {
    const loadSymptoms = async () => {
      try {
        const response = await fetch('/symptoms.txt');
        if (!response.ok) {
          throw new Error('Failed to load symptoms file');
        }
        const text = await response.text();
        const symptoms = text
          .split('\n')
          .map(symptom => symptom.trim())
          .filter(symptom => symptom.length > 0);

        setAllSymptoms(symptoms);
      } catch (err) {
        console.error('Error loading symptoms:', err);
        setFileError('Failed to load symptoms list. Please try again later.');
      }
    };

    loadSymptoms();
  }, []);

  const handleInputChange = (e) => {
    const value = e.target.value;
    setInputValue(value);

    if (value.trim()) {
      const filtered = allSymptoms.filter(
        symptom =>
          symptom.toLowerCase().includes(value.toLowerCase()) &&
          !symptoms.includes(symptom)
      );
      setSuggestions(filtered);
    } else {
      setSuggestions([]);
    }
  };

  const addSymptom = (symptom) => {
    if (!symptoms.includes(symptom)) {
      setSymptoms([...symptoms, symptom]);
      setInputValue('');
      setSuggestions([]);
    }
  };

  const removeSymptom = (symptomToRemove) => {
    setSymptoms(symptoms.filter(symptom => symptom !== symptomToRemove));
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && suggestions.length > 0) {
      addSymptom(suggestions[0]);
    }
  };

  const resetFields = () => {
    setInputValue('');
    setSymptoms([]);
    setSuggestions([]);
    setPrediction(null);
    setError(null);
  };

  const getPrediction = async () => {
    if (symptoms.length === 0) {
      setError('Please add at least one symptom');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const payload = { symptoms };

      if (isAuthenticated && user?.id) {
        payload.user_id = user.id;
        console.log('Adding user_id to payload:', user.id);
      } else {
        console.log('User not authenticated, skipping user_id');
      }

      const token = localStorage.getItem('token');

      console.log('Sending prediction request with payload:', payload);
      console.log('Authentication status:', isAuthenticated, 'Token exists:', !!token);

      const response = await fetch(`${BASE_URL}/prediction/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token && { Authorization: `Bearer ${token}` }),
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('API error response:', errorText);
        throw new Error(`Prediction failed: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      console.log('Prediction API response:', data);

      setPrediction(data);

      if (onPredictionResult && typeof onPredictionResult === 'function') {
        console.log('Passing prediction to parent component:', data);
        onPredictionResult(data);
      }
    } catch (err) {
      console.error('Prediction error:', err);
      setError(`Failed to get prediction: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  if (fileError) {
    return (
      <div className="max-w-2xl mx-auto p-6">
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          <p>{fileError}</p>
        </div>
      </div>
    );
  }

  // Helper function to render the final prediction
  const renderFinalPrediction = () => {
    if (!prediction || !prediction.final_prediction) return null;
    
    if (Array.isArray(prediction.final_prediction)) {
      return (
        <div className="mt-1">
          {prediction.final_prediction.map((disease, index) => (
            <div key={index} className="bg-blue-50 text-blue-800 px-3 py-2 rounded mb-1">
              {disease}
            </div>
          ))}
        </div>
      );
    } else {
      // Handle single disease case
      return (
        <div className="mt-1">
          <div className="bg-blue-50 text-blue-800 px-3 py-2 rounded">
            {prediction.final_prediction}
          </div>
        </div>
      );
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6 relative">
      {/* Blurred Overlay when loading */}
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50 backdrop-blur-md rounded-lg pointer-events-none">
          <LifeLine color="#ffffff" size="medium" text="Predicting..." />
        </div>
      )}

      <h2 className="text-2xl font-bold mb-4">Enter Symptoms</h2>

      <div className={`border rounded-lg p-2 mb-4 ${loading ? 'blur-sm' : ''}`}>
        <div className="flex flex-wrap gap-2 mb-2">
          {symptoms.map((symptom) => (
            <span key={symptom} className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full flex items-center gap-1">
              {symptom}
              <button onClick={() => removeSymptom(symptom)} className="hover:text-blue-600">
                <X size={14} />
              </button>
            </span>
          ))}
        </div>

        <div className="flex items-center">
          <input
            type="text"
            value={inputValue}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder="Type symptom..."
            className="w-full p-2 outline-none border rounded-lg fade-placeholder text-gray-800"
          />
          <button onClick={resetFields} className="ml-2 bg-gray-200 text-gray-800 px-3 py-1 rounded hover:bg-gray-300">
            Reset
          </button>
        </div>
      </div>

      {suggestions.length > 0 && (
        <div className="border rounded-lg mt-1 shadow-lg">
          {suggestions.map((suggestion) => (
            <button key={suggestion} onClick={() => addSymptom(suggestion)} className="w-full text-left px-4 py-2 hover:bg-gray-100 mb-1 rounded">
              {suggestion}
            </button>
          ))}
        </div>
      )}

      {!isAuthenticated && (
        <div className="mb-4 text-sm text-gray-600 border-l-4 border-blue-500 pl-3 py-2 bg-blue-50">
          <p>Sign in to save your prediction history.</p>
        </div>
      )}

      <button
        onClick={getPrediction}
        disabled={loading || symptoms.length === 0}
        className="w-full bg-gradient-to-r from-[#ff8c42] to-[#ff3e55] text-white font-bold py-2 px-6 rounded-lg hover:scale-105 transition-all duration-300"
      >
        {loading ? 'Predicting...' : 'Predict Disease'}
      </button>

      {error && <div className="mt-4 text-red-600">{error}</div>}

      {prediction && (
        <div className="mt-6 border rounded-lg p-4">
          <h3 className="font-bold mb-3">Prediction Results</h3>
          <p className="font-semibold">Final Prediction:</p>
          {renderFinalPrediction()}
        </div>
      )}
    </div>
  );
};

export default SymptomPredictor;