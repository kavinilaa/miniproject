package com.medical.auth;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "patient_reports")
public class PatientReport {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    // ── Core ──────────────────────────────────────────────────────────────────
    @Column(name = "doctor_id",    nullable = false) private String doctorId;
    @Column(name = "patient_id")                     private String patientId;
    @Column(name = "patient_name", nullable = false) private String patientName;
    @Column(name = "age",          nullable = false) private int    age;
    @Column(name = "gender",       nullable = false) private String gender;
    @Column(name = "height")                         private Double height;
    @Column(name = "weight")                         private Double weight;
    @Column(name = "bmi")                            private Double bmi;
    @Column(name = "blood_group")                    private String bloodGroup;
    @Column(name = "ethnicity")                      private String ethnicity;

    // ── Medical History ───────────────────────────────────────────────────────
    @Column(name = "prev_colorectal_surgery") private boolean prevColorectalSurgery;
    @Column(name = "family_history_polyps")   private boolean familyHistoryPolyps;
    @Column(name = "ibd")                     private boolean ibd;
    @Column(name = "genetic_syndrome")        private boolean geneticSyndrome;
    @Column(name = "chronic_constipation")    private boolean chronicConstipation;
    @Column(name = "smoking")                 private boolean smoking;

    // ── Lifestyle ─────────────────────────────────────────────────────────────
    @Column(name = "diet_type")        private String dietType;
    @Column(name = "activity_level")   private String activityLevel;
    @Column(name = "sleep_quality")    private String sleepQuality;
    @Column(name = "stress_level")     private String stressLevel;

    // ── Clinical Parameters ───────────────────────────────────────────────────
    @Column(name = "hemoglobin")   private Double hemoglobin;
    @Column(name = "blood_sugar")  private Double bloodSugar;
    @Column(name = "crp")          private Double crp;
    @Column(name = "cholesterol")  private Double cholesterol;

    // ── Symptoms ──────────────────────────────────────────────────────────────
    @Column(name = "symptoms", columnDefinition = "TEXT") private String symptoms;

    // ── Colonoscopy Details ───────────────────────────────────────────────────
    @Column(name = "polyp_location")       private String polypLocation;
    @Column(name = "polyp_shape")          private String polypShape;
    @Column(name = "surface_texture")      private String surfaceTexture;
    @Column(name = "polyp_color")          private String polypColor;

    // ── AI Result ─────────────────────────────────────────────────────────────
    @Column(name = "prediction")                          private String prediction;
    @Column(name = "confidence")                          private double confidence;
    @Column(name = "risk_score")                          private String riskScore;
    @Column(name = "image_path")                          private String imagePath;
    @Column(name = "heatmap", columnDefinition = "LONGTEXT") private String heatmap;
    @Column(name = "date")                                private LocalDateTime date;

    @PrePersist
    public void prePersist() { this.date = LocalDateTime.now(); }

    // ── Getters & Setters ─────────────────────────────────────────────────────
    public Long getId()                              { return id; }
    public String getDoctorId()                      { return doctorId; }
    public void setDoctorId(String v)                { this.doctorId = v; }
    public String getPatientId()                     { return patientId; }
    public void setPatientId(String v)               { this.patientId = v; }
    public String getPatientName()                   { return patientName; }
    public void setPatientName(String v)             { this.patientName = v; }
    public int getAge()                              { return age; }
    public void setAge(int v)                        { this.age = v; }
    public String getGender()                        { return gender; }
    public void setGender(String v)                  { this.gender = v; }
    public Double getHeight()                        { return height; }
    public void setHeight(Double v)                  { this.height = v; }
    public Double getWeight()                        { return weight; }
    public void setWeight(Double v)                  { this.weight = v; }
    public Double getBmi()                           { return bmi; }
    public void setBmi(Double v)                     { this.bmi = v; }
    public String getBloodGroup()                    { return bloodGroup; }
    public void setBloodGroup(String v)              { this.bloodGroup = v; }
    public String getEthnicity()                     { return ethnicity; }
    public void setEthnicity(String v)               { this.ethnicity = v; }
    public boolean isPrevColorectalSurgery()         { return prevColorectalSurgery; }
    public void setPrevColorectalSurgery(boolean v)  { this.prevColorectalSurgery = v; }
    public boolean isFamilyHistoryPolyps()           { return familyHistoryPolyps; }
    public void setFamilyHistoryPolyps(boolean v)    { this.familyHistoryPolyps = v; }
    public boolean isIbd()                           { return ibd; }
    public void setIbd(boolean v)                    { this.ibd = v; }
    public boolean isGeneticSyndrome()               { return geneticSyndrome; }
    public void setGeneticSyndrome(boolean v)        { this.geneticSyndrome = v; }
    public boolean isChronicConstipation()           { return chronicConstipation; }
    public void setChronicConstipation(boolean v)    { this.chronicConstipation = v; }
    public boolean isSmoking()                       { return smoking; }
    public void setSmoking(boolean v)                { this.smoking = v; }
    public String getDietType()                      { return dietType; }
    public void setDietType(String v)                { this.dietType = v; }
    public String getActivityLevel()                 { return activityLevel; }
    public void setActivityLevel(String v)           { this.activityLevel = v; }
    public String getSleepQuality()                  { return sleepQuality; }
    public void setSleepQuality(String v)            { this.sleepQuality = v; }
    public String getStressLevel()                   { return stressLevel; }
    public void setStressLevel(String v)             { this.stressLevel = v; }
    public Double getHemoglobin()                    { return hemoglobin; }
    public void setHemoglobin(Double v)              { this.hemoglobin = v; }
    public Double getBloodSugar()                    { return bloodSugar; }
    public void setBloodSugar(Double v)              { this.bloodSugar = v; }
    public Double getCrp()                           { return crp; }
    public void setCrp(Double v)                     { this.crp = v; }
    public Double getCholesterol()                   { return cholesterol; }
    public void setCholesterol(Double v)             { this.cholesterol = v; }
    public String getSymptoms()                      { return symptoms; }
    public void setSymptoms(String v)                { this.symptoms = v; }
    public String getPolypLocation()                 { return polypLocation; }
    public void setPolypLocation(String v)           { this.polypLocation = v; }
    public String getPolypShape()                    { return polypShape; }
    public void setPolypShape(String v)              { this.polypShape = v; }
    public String getSurfaceTexture()                { return surfaceTexture; }
    public void setSurfaceTexture(String v)          { this.surfaceTexture = v; }
    public String getPolypColor()                    { return polypColor; }
    public void setPolypColor(String v)              { this.polypColor = v; }
    public String getPrediction()                    { return prediction; }
    public void setPrediction(String v)              { this.prediction = v; }
    public double getConfidence()                    { return confidence; }
    public void setConfidence(double v)              { this.confidence = v; }
    public String getRiskScore()                     { return riskScore; }
    public void setRiskScore(String v)               { this.riskScore = v; }
    public String getImagePath()                     { return imagePath; }
    public void setImagePath(String v)               { this.imagePath = v; }
    public String getHeatmap()                       { return heatmap; }
    public void setHeatmap(String v)                 { this.heatmap = v; }
    public LocalDateTime getDate()                   { return date; }
    public void setDate(LocalDateTime v)             { this.date = v; }
}
