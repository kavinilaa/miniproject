import { useState } from "react";
import "../styles/Register.css";

const EMAIL_REGEX = /^[A-Za-z0-9+_.-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$/;

export default function Register({ onNavigateLogin }) {
  const [form, setForm] = useState({
    doctorName: "",
    doctorId: "",
    email: "",
    password: "",
    confirmPassword: "",
  });
  const [errors, setErrors] = useState({});
  const [success, setSuccess] = useState("");
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
    setErrors({ ...errors, [e.target.name]: "" });
  };

  const validate = () => {
    const newErrors = {};
    if (!form.doctorName.trim())       newErrors.doctorName = "Doctor name is required.";
    if (!form.doctorId.trim())         newErrors.doctorId = "Doctor ID is required.";
    if (!form.email.trim())            newErrors.email = "Email is required.";
    else if (!EMAIL_REGEX.test(form.email)) newErrors.email = "Invalid email format.";
    if (!form.password)                newErrors.password = "Password is required.";
    if (!form.confirmPassword)         newErrors.confirmPassword = "Please confirm your password.";
    else if (form.password !== form.confirmPassword)
                                       newErrors.confirmPassword = "Passwords do not match.";
    return newErrors;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSuccess("");
    const validationErrors = validate();
    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors);
      return;
    }

    setLoading(true);
    try {
      const res = await fetch("http://localhost:8080/api/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          doctorName: form.doctorName,
          doctorId: form.doctorId,
          email: form.email,
          password: form.password,
        }),
      });

      const data = await res.json();
      if (res.ok) {
        setSuccess("Registration successful! You can now log in.");
        setForm({ doctorName: "", doctorId: "", email: "", password: "", confirmPassword: "" });
      } else {
        setErrors({ api: data.message || "Registration failed." });
      }
    } catch {
      setErrors({ api: "Unable to connect to server." });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="register-wrapper">
      <div className="register-card">
        <div className="register-header">
          <span className="register-icon">🏥</span>
          <h1>MediCare Portal</h1>
          <p>Doctor Registration</p>
        </div>

        <form onSubmit={handleSubmit} noValidate>
          {[
            { name: "doctorName",      label: "Doctor Name",      type: "text",     placeholder: "Enter full name" },
            { name: "doctorId",        label: "Doctor ID",        type: "text",     placeholder: "Enter Doctor ID" },
            { name: "email",           label: "Email",            type: "email",    placeholder: "Enter email address" },
            { name: "password",        label: "Password",         type: "password", placeholder: "Create a password" },
            { name: "confirmPassword", label: "Confirm Password", type: "password", placeholder: "Repeat your password" },
          ].map(({ name, label, type, placeholder }) => (
            <div className="form-group" key={name}>
              <label htmlFor={name}>{label}</label>
              <input
                id={name}
                name={name}
                type={type}
                placeholder={placeholder}
                value={form[name]}
                onChange={handleChange}
                className={errors[name] ? "input-error" : ""}
              />
              {errors[name] && <span className="field-error">⚠ {errors[name]}</span>}
            </div>
          ))}

          {errors.api && <p className="error-msg">⚠ {errors.api}</p>}
          {success  && <p className="success-msg">✔ {success}</p>}

          <button type="submit" className="submit-btn" disabled={loading}>
            {loading ? "Registering..." : "Register"}
          </button>
        </form>

        <p className="nav-link">
          Already have an account?{" "}
          <span onClick={onNavigateLogin}>Login here</span>
        </p>
      </div>
    </div>
  );
}
