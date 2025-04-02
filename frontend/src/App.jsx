import { useState } from "react";
import "./App.css";
import { Route, Routes } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";

// Import existing components
import HomePage from "./components/mainHome";
import Login from "./components/login";
import Signup from "./components/signup";
import Squares from "./components/Squares";
import SymptomPredictor from "./components/SymptomPredictor";
import Predict from "./components/Predict";
import HospitalList from "./components/Hospitals";
import ChatBot from "./components/ChatBot";

// Import or create new components for additional routes
// import About from "./components/About"; // Create this component
// import Contact from "./components/Contact"; // Create this component
// import Doctors from "./components/Doctors"; // Create this component
// import PreviousPredictions from "./components/PreviousPredictions"; // Create this component
import Navbar from "./components/Navbar"; // Your updated Navbar component
import AboutUs from "./components/AboutUs";
import ContactUsPage from "./components/ContactUsPage";
import DoctorList from "./components/DoctorList";
import PreviousPrediction from "./components/PreviousPrediction";
import Appointments from "./components/Appointment";

function App() {
  const [count, setCount] = useState(0);

  return (
    <AuthProvider>
      <div className="App">
        <Navbar />
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/hospital" element={<HospitalList />} />
          <Route path="/aboutus" element={<AboutUs />} />
          <Route path="/contact" element={<ContactUsPage />} />
          <Route path="/doctors" element={<DoctorList />} />
          <Route path="/previous-predictions" element={<PreviousPrediction />} />
          <Route path="/appointments" element={<Appointments />} />
        </Routes>
        {/* Uncomment if you want the ChatBot on all pages */}
        {/* <ChatBot /> */}
      </div>
    </AuthProvider>
  );
}

export default App;