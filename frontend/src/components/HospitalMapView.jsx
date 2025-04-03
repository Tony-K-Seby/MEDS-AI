import React, { useEffect, useState } from "react";
import { Map, Marker, Overlay } from "pigeon-maps";

const HospitalMapView = () => {
  const [hospitals, setHospitals] = useState([]);

  useEffect(() => {
    fetch("http://127.0.0.1:5000/hospitals/")
      .then((response) => response.json())
      .then((data) => setHospitals(data))
      .catch((error) => console.error("Error fetching hospitals:", error));
  }, []);

  return (
    <div className="w-full h-full rounded-xl shadow-lg overflow-hidden">
      <Map defaultCenter={[10.52, 76.21]} defaultZoom={12} height={500}>
        {hospitals.map((hospital) => (
          <Marker
            key={hospital._id}
            width={40}
            anchor={[hospital.location.latitude, hospital.location.longitude]}
          />
        ))}

        {hospitals.map((hospital) => (
          <Overlay
            key={hospital._id}
            anchor={[hospital.location.latitude, hospital.location.longitude]}
            offset={[0, 20]}
          >
            <div className="bg-white p-2 rounded-md shadow-lg text-xs text-black">
              <strong>{hospital.name}</strong>
              <p>{hospital.address}</p>
              <p>📞 {hospital.contact_no}</p>
            </div>
          </Overlay>
        ))}
      </Map>
    </div>
  );
};

export default HospitalMapView;
