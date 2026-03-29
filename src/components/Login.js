import { useState } from "react";
import "../styles/Login.css";

export default function Login({ onNavigateRegister, onLoginSuccess }) {
  const [doctorId, setDoctorId] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError]     = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (!doctorId.trim() || !password.trim()) {
      setError("All fields are required.");
      return;
    }

    setLoading(true);
    try {
      const res = await fetch("http://localhost:8080/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doctorId, password }),
      });

      const data = await res.json();
      if (res.ok) {
        localStorage.setItem("token", data.token);
        localStorage.setItem("doctorId", data.doctorId);
        onLoginSuccess({ doctorId: data.doctorId, fullName: data.fullName, email: data.email });
      } else {
        setError(data.message || "Login failed.");
      }
    } catch {
      setError("Unable to connect to server.");
    } finally {
      setLoading(false);
    }
  };

return (
    <div className="login-wrapper">
      <div className="login-card">
        <div className="login-header">
          <span className="login-icon">🏥</span>
          <h1>MediCare Portal</h1>
          <p>Doctor Login</p>
        </div>

        <form onSubmit={handleSubmit} noValidate>
          <div className="form-group">
            <label htmlFor="doctorId">Doctor ID</label>
            <input
              id="doctorId"
              type="text"
              placeholder="Enter your Doctor ID"
              value={doctorId}
              onChange={(e) => setDoctorId(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              placeholder="Enter your password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>

          {error && <p className="error-msg">⚠ {error}</p>}

          <button type="submit" className="submit-btn" disabled={loading}>
            {loading ? "Logging in..." : "Login"}
          </button>
        </form>

        <p className="nav-link">
          Don't have an account?{" "}
          <span onClick={onNavigateRegister}>Register here</span>
        </p>
      </div>
    </div>
  );
}
