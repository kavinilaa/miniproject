import { useState, useEffect, useCallback, useMemo } from "react";
import "../styles/PatientForm.css";

const TABS = [
  { label: "Basic", icon: "ID", desc: "Identity and demographics" },
  { label: "Medical History", icon: "HX", desc: "Risk-linked history" },
  { label: "Lifestyle", icon: "LF", desc: "Habits and wellbeing" },
  { label: "Clinical", icon: "LB", desc: "Lab parameters" },
  { label: "Symptoms", icon: "SX", desc: "Observed symptoms" },
  { label: "Colonoscopy", icon: "SC", desc: "Procedure observations" },
];

const SYMPTOM_OPTIONS = [
  "Rectal bleeding", "Abdominal pain", "Weight loss",
  "Fatigue", "Change in bowel habits", "Nausea", "Bloating",
];

const INITIAL = {
  // Basic
  patientId: "", patientName: "", age: "", gender: "",
  height: "", weight: "", bmi: "", bloodGroup: "", ethnicity: "",
  // Medical history
  prevColorectalSurgery: false, familyHistoryPolyps: false,
  ibd: false, geneticSyndrome: false, chronicConstipation: false, smoking: false,
  // Lifestyle
  dietType: "", activityLevel: "", sleepQuality: "", stressLevel: "",
  // Clinical
  hemoglobin: "", bloodSugar: "", crp: "", cholesterol: "",
  // Symptoms
  symptoms: [],
  // Colonoscopy
  polypLocation: "", polypShape: "", surfaceTexture: "", polypColor: "",
  clinicalRisk: "Low",
};

export default function PatientForm({ value, onChange, onSave, saved, error }) {
  const [tab, setTab]   = useState(0);
  const [bmi, setBmi]   = useState(null);
  const [risk, setRisk] = useState(null);

  const p  = value;
  const up = useCallback((field, val) => {
    onChange((prev) => ({ ...prev, [field]: val }));
  }, [onChange]);

  // Auto BMI
  useEffect(() => {
    const h = parseFloat(p.height), w = parseFloat(p.weight);
    if (h > 0 && w > 0) {
      const b = (w / Math.pow(h / 100, 2)).toFixed(1);
      setBmi(b);
      if (String(p.bmi || "") !== b) up("bmi", b);
    } else {
      setBmi(null);
      if (p.bmi) up("bmi", "");
    }
  }, [p.height, p.weight, p.bmi, up]);

  // Auto risk score
  useEffect(() => {
    const age = parseInt(p.age);
    if (!age && !p.smoking && !p.familyHistoryPolyps && !p.ibd &&
        !p.geneticSyndrome && !p.crp && !p.bloodSugar && !bmi) {
      setRisk(null);
      return;
    }
    let score = 0;
    if (age >= 50) score += 2; else if (age >= 40) score += 1;
    if (p.smoking)               score += 2;
    if (p.familyHistoryPolyps)   score += 3;
    if (p.ibd)                   score += 2;
    if (p.geneticSyndrome)       score += 3;
    if (parseFloat(p.crp) > 5)   score += 2;
    if (parseFloat(p.bloodSugar) > 126) score += 1;
    if (bmi && parseFloat(bmi) >= 30)   score += 1;
    let computedRisk = "Low";
    if (score >= 7) computedRisk = "High";
    else if (score >= 4) computedRisk = "Medium";

    setRisk(computedRisk);
    if ((p.clinicalRisk || "") !== computedRisk) up("clinicalRisk", computedRisk);
  }, [p.age, p.smoking, p.familyHistoryPolyps, p.ibd, p.geneticSyndrome, p.crp, p.bloodSugar, p.clinicalRisk, bmi, up]);

  const toggleSymptom = (s) => {
    const list = p.symptoms.includes(s)
      ? p.symptoms.filter((x) => x !== s)
      : [...p.symptoms, s];
    up("symptoms", list);
  };

  const riskColor = { High: "#f87171", Medium: "#fbbf24", Low: "#4ade80" }[risk] || "#a78bfa";

  const completion = useMemo(() => {
    const keys = [
      "patientName", "age", "gender", "height", "weight",
      "dietType", "activityLevel", "hemoglobin", "bloodSugar",
      "crp", "cholesterol", "polypLocation", "polypShape",
    ];
    const filled = keys.filter((k) => String(p[k] || "").trim() !== "").length;
    return Math.round((filled / keys.length) * 100);
  }, [p]);

  return (
    <div className="pf-wrapper">
      <div className="pf-hero">
        <div>
          <p className="pf-kicker">Patient Intelligence Intake</p>
          <h3 className="pf-title">Structured Clinical Capture</h3>
          <p className="pf-subtitle">Complete all sections for stronger multimodal prediction quality.</p>
        </div>
        <div className="pf-progress">
          <span>Completion</span>
          <strong>{completion}%</strong>
          <div className="pf-progress-track">
            <div className="pf-progress-fill" style={{ width: `${completion}%` }} />
          </div>
        </div>
      </div>

      {/* Risk Score Banner */}
      {risk && (
        <div className="risk-banner" style={{ borderColor: riskColor, color: riskColor }}>
          <span className="risk-icon">
            {risk === "High" ? "🔴" : risk === "Medium" ? "🟡" : "🟢"}
          </span>
          <div>
            <p className="risk-label">Clinical Risk Score</p>
            <p className="risk-value">{risk} Risk</p>
          </div>
          {bmi && (
            <div className="bmi-box">
              <p className="risk-label">BMI</p>
              <p className="risk-value">{bmi}</p>
            </div>
          )}
        </div>
      )}

      {/* Tabs */}
      <div className="pf-tabs">
        {TABS.map((t, i) => (
          <button key={t.label} className={`pf-tab ${tab === i ? "active" : ""}`} onClick={() => setTab(i)}>
            <span className="tab-step">{i + 1}</span>
            <span className="tab-icon">{t.icon}</span>
            <span className="tab-text">
              <strong>{t.label}</strong>
              <small>{t.desc}</small>
            </span>
          </button>
        ))}
      </div>

      <div className="pf-body">
        <div className="pf-body-head">
          <h4>{TABS[tab].label}</h4>
          <p>{TABS[tab].desc}</p>
        </div>

        {/* ── BASIC ── */}
        {tab === 0 && (
          <div className="pf-grid">
            <Field label="Patient ID"   value={p.patientId}   onChange={(v) => up("patientId", v)}   placeholder="e.g. PAT-001" />
            <Field label="Patient Name" value={p.patientName} onChange={(v) => up("patientName", v)} placeholder="Full name" required />
            <Field label="Age"          value={p.age}         onChange={(v) => up("age", v)}          placeholder="Years" type="number" required />
            <SelectField label="Gender" value={p.gender} onChange={(v) => up("gender", v)} required
              options={["Male", "Female", "Other"]} />
            <Field label="Height (cm)"  value={p.height}      onChange={(v) => up("height", v)}       placeholder="e.g. 170" type="number" />
            <Field label="Weight (kg)"  value={p.weight}      onChange={(v) => up("weight", v)}       placeholder="e.g. 70"  type="number" />
            <SelectField label="Blood Group" value={p.bloodGroup} onChange={(v) => up("bloodGroup", v)}
              options={["A+","A-","B+","B-","AB+","AB-","O+","O-"]} />
            <SelectField label="Ethnicity" value={p.ethnicity} onChange={(v) => up("ethnicity", v)}
              options={["Asian","Caucasian","African","Hispanic","Middle Eastern","Other"]} />
            {bmi && (
              <div className="bmi-display">
                <span>Auto BMI:</span>
                <strong style={{ color: parseFloat(bmi) >= 30 ? "#f87171" : parseFloat(bmi) >= 25 ? "#fbbf24" : "#4ade80" }}>
                  {bmi} {parseFloat(bmi) >= 30 ? "(Obese)" : parseFloat(bmi) >= 25 ? "(Overweight)" : "(Normal)"}
                </strong>
              </div>
            )}
          </div>
        )}

        {/* ── MEDICAL HISTORY ── */}
        {tab === 1 && (
          <div className="pf-checks">
            {[
              ["prevColorectalSurgery", "Previous colorectal surgery"],
              ["familyHistoryPolyps",   "Family history of polyps"],
              ["ibd",                   "Inflammatory bowel disease (IBD)"],
              ["geneticSyndrome",       "Genetic syndrome (FAP / Lynch)"],
              ["chronicConstipation",   "Chronic constipation"],
              ["smoking",               "Smoking"],
            ].map(([key, label]) => (
              <label key={key} className="check-row">
                <input type="checkbox" checked={p[key]} onChange={(e) => up(key, e.target.checked)} />
                <span className="check-label">{label}</span>
                {(key === "familyHistoryPolyps" || key === "geneticSyndrome") && (
                  <span className="high-risk-tag">High Risk Factor</span>
                )}
              </label>
            ))}
          </div>
        )}

        {/* ── LIFESTYLE ── */}
        {tab === 2 && (
          <div className="pf-grid">
            <SelectField label="Diet Type" value={p.dietType} onChange={(v) => up("dietType", v)}
              options={["High Fat","Fiber Rich","Balanced","Vegetarian","Vegan","Processed Food"]} />
            <SelectField label="Physical Activity" value={p.activityLevel} onChange={(v) => up("activityLevel", v)}
              options={["Sedentary","Light","Moderate","Active","Very Active"]} />
            <SelectField label="Sleep Quality" value={p.sleepQuality} onChange={(v) => up("sleepQuality", v)}
              options={["Poor","Fair","Good","Excellent"]} />
            <SelectField label="Stress Level" value={p.stressLevel} onChange={(v) => up("stressLevel", v)}
              options={["Low","Moderate","High","Very High"]} />
          </div>
        )}

        {/* ── CLINICAL PARAMETERS ── */}
        {tab === 3 && (
          <div className="pf-grid">
            <Field label="Hemoglobin (g/dL)"  value={p.hemoglobin}  onChange={(v) => up("hemoglobin", v)}  placeholder="Normal: 12–17" type="number" />
            <Field label="Blood Sugar (mg/dL)" value={p.bloodSugar}  onChange={(v) => up("bloodSugar", v)}  placeholder="Normal: <100"  type="number" />
            <Field label="CRP (mg/L)"          value={p.crp}         onChange={(v) => up("crp", v)}         placeholder="Normal: <5"    type="number" />
            <Field label="Cholesterol (mg/dL)" value={p.cholesterol} onChange={(v) => up("cholesterol", v)} placeholder="Normal: <200"  type="number" />
            <div className="clinical-hints">
              <p>🔬 <strong>CRP &gt; 5</strong> indicates inflammation — increases risk score</p>
              <p>🩸 <strong>Blood Sugar &gt; 126</strong> indicates diabetes — increases risk score</p>
              <p>⚖️ <strong>BMI ≥ 30</strong> (Obese) — increases risk score</p>
            </div>
          </div>
        )}

        {/* ── SYMPTOMS ── */}
        {tab === 4 && (
          <div className="pf-symptoms">
            <p className="symptoms-hint">Select all symptoms that apply:</p>
            <div className="symptom-grid">
              {SYMPTOM_OPTIONS.map((s) => (
                <button key={s}
                  className={`symptom-chip ${p.symptoms.includes(s) ? "selected" : ""}`}
                  onClick={() => toggleSymptom(s)}>
                  {p.symptoms.includes(s) ? "✔ " : ""}{s}
                </button>
              ))}
            </div>
            {p.symptoms.length > 0 && (
              <div className="selected-symptoms">
                <p>Selected: {p.symptoms.join(", ")}</p>
              </div>
            )}
          </div>
        )}

        {/* ── COLONOSCOPY DETAILS ── */}
        {tab === 5 && (
          <div className="pf-grid">
            <SelectField label="Polyp Location" value={p.polypLocation} onChange={(v) => up("polypLocation", v)}
              options={["Ascending Colon","Transverse Colon","Descending Colon","Sigmoid Colon","Rectum","Cecum"]} />
            <SelectField label="Polyp Shape" value={p.polypShape} onChange={(v) => up("polypShape", v)}
              options={["Flat","Pedunculated","Sessile","Semi-pedunculated"]} />
            <SelectField label="Surface Texture" value={p.surfaceTexture} onChange={(v) => up("surfaceTexture", v)}
              options={["Smooth","Irregular","Lobulated","Villous"]} />
            <SelectField label="Polyp Color" value={p.polypColor} onChange={(v) => up("polypColor", v)}
              options={["Pink","Red","White","Dark Red","Pale"]} />
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="pf-footer">
        <div className="pf-nav">
          {tab > 0 && <button className="btn-secondary" onClick={() => setTab(tab - 1)}>← Previous</button>}
          {tab < TABS.length - 1 && <button className="btn-primary" onClick={() => setTab(tab + 1)}>Next →</button>}
        </div>
        {error  && <p className="error-msg">⚠ {error}</p>}
        {saved  && <p className="success-msg">✔ Patient info saved. Go to Upload tab to analyze.</p>}
        <button className="btn-save" onClick={onSave}>💾 Save Patient Info</button>
      </div>
    </div>
  );
}

function Field({ label, value, onChange, placeholder, type = "text", required }) {
  return (
    <div className="pf-field">
      <label>{label}{required && <span className="req">*</span>}</label>
      <input type={type} value={value} placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)} />
    </div>
  );
}

function SelectField({ label, value, onChange, options, required }) {
  return (
    <div className="pf-field">
      <label>{label}{required && <span className="req">*</span>}</label>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="">Select...</option>
        {options.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  );
}

export { INITIAL };
