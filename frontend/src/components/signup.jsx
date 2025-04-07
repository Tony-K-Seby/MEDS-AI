import { useState } from "react";
import caduceus from "../assets/caduceus.png";

export default function Signup() {

  const BASE_URL = import.meta.env.VITE_BASE_URL;

  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
    gender: "",
    age: "",
    address: "",
    contact_no: "",
  });

  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  // Handle input changes
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: value,
    }));
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setSuccess("");

    try {
      const response = await fetch(`${BASE_URL}/auth/register`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Registration failed");
      }

      setSuccess("User registered successfully!");
      setFormData({
        name: "",
        email: "",
        password: "",
        gender: "",
        age: "",
        address: "",
        contact_no: "",
      });
    } catch (error) {
      setError(error.message);
    }
  };

  return (
    <div className="flex h-screen w-screen">
      {/* Left Side - Image */}
      <div className="hidden lg:flex lg:w-1/2 bg-gray-100 items-center justify-center">
        <img
          src={caduceus}
          alt="Signup"
          className="h-full w-full object-cover"
        />
      </div>

      {/* Right Side - Form */}
      <div className="flex w-full lg:w-1/2 items-center justify-center p-4 bg-white">
        <div className="max-w-sm w-full p-3 border border-gray-200 rounded-md shadow-md">
          <h2 className="text-xl font-semibold text-gray-900 text-center mb-3">
            Create an account
          </h2>

          {error && <p className="text-red-500 text-sm text-center">{error}</p>}
          {success && <p className="text-green-500 text-sm text-center">{success}</p>}

          <form onSubmit={handleSubmit}>
            <div className="grid grid-cols-1 gap-y-2">
              {[
                { label: "Name", name: "name", type: "text" },
                { label: "Email", name: "email", type: "email" },
                { label: "Password", name: "password", type: "password" },
                { label: "Age", name: "age", type: "number", min: 1, max: 120 },
                { label: "Address", name: "address", type: "textarea" },
                { label: "Phone number", name: "contact_no", type: "text" },
              ].map(({ label, name, type, ...rest }) => (
                <div key={name}>
                  <label className="block text-xs font-bold text-gray-900">{label}</label>
                  {type === "textarea" ? (
                    <textarea
                      name={name}
                      rows="2"
                      value={formData[name]}
                      onChange={handleChange}
                      className="mt-1 block w-full rounded border border-gray-300 px-2 py-1 text-gray-900 shadow-sm placeholder-gray-400 focus:border-indigo-600 focus:ring-indigo-600 text-sm"
                      {...rest}
                    />
                  ) : (
                    <input
                      type={type}
                      name={name}
                      value={formData[name]}
                      onChange={handleChange}
                      className="mt-1 block w-full rounded border border-gray-300 px-2 py-1 text-gray-900 shadow-sm placeholder-gray-400 focus:border-indigo-600 focus:ring-indigo-600 text-sm"
                      {...rest}
                    />
                  )}
                </div>
              ))}

              {/* Gender Selection */}
              <div>
                <label className="block text-xs font-bold text-gray-900">Gender</label>
                <div className="mt-1 flex space-x-2">
                  {["male", "female"].map((gender) => (
                    <label key={gender} className="inline-flex items-center text-sm">
                      <input
                        type="radio"
                        name="gender"
                        value={gender}
                        checked={formData.gender === gender}
                        onChange={handleChange}
                        className="h-3 w-3 border-gray-300 text-indigo-600 focus:ring-indigo-600"
                      />
                      <span className="ml-1 text-gray-900">{gender.charAt(0).toUpperCase() + gender.slice(1)}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div className="mt-2">
                <button
                  type="submit"
                  className="w-full rounded bg-[#FF6F00] px-3 py-1.5 text-white font-semibold shadow hover:bg-[#D65C00] focus:outline-none focus:ring-2 focus:ring-indigo-600 focus:ring-offset-2 text-sm"
                >
                  Create Account
                </button>
              </div>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
