import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import Login from "./components/login.jsx";
import Signup from "./components/signup.jsx";
import Home from "./components/mainHome.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <Home />
    {/* <Signup /> */}
    {/* <Login /> */}
  </StrictMode>
);
