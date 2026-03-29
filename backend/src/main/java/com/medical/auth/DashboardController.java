package com.medical.auth;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/dashboard")
@CrossOrigin(origins = "http://localhost:3000")
public class DashboardController {

    private final PatientReportService reportService;
    private final JwtUtil              jwtUtil;
    private final DoctorService        doctorService;
    private final FlaskService         flaskService;

    public DashboardController(PatientReportService reportService, JwtUtil jwtUtil,
                               DoctorService doctorService, FlaskService flaskService) {
        this.reportService = reportService;
        this.jwtUtil       = jwtUtil;
        this.doctorService = doctorService;
        this.flaskService  = flaskService;
    }

    @PostMapping("/predict")
    public ResponseEntity<?> predict(
            @RequestHeader("Authorization") String authHeader,
            @RequestParam("image")                  MultipartFile image,
            // Basic
            @RequestParam("patientId")              String patientId,
            @RequestParam("patientName")            String patientName,
            @RequestParam("age")                    int    age,
            @RequestParam("gender")                 String gender,
            @RequestParam(value="height",           required=false) Double height,
            @RequestParam(value="weight",           required=false) Double weight,
            @RequestParam(value="bloodGroup",       required=false) String bloodGroup,
            @RequestParam(value="ethnicity",        required=false) String ethnicity,
            // Medical history
            @RequestParam(value="prevColorectalSurgery", defaultValue="false") boolean prevColorectalSurgery,
            @RequestParam(value="familyHistoryPolyps",   defaultValue="false") boolean familyHistoryPolyps,
            @RequestParam(value="ibd",                   defaultValue="false") boolean ibd,
            @RequestParam(value="geneticSyndrome",       defaultValue="false") boolean geneticSyndrome,
            @RequestParam(value="chronicConstipation",   defaultValue="false") boolean chronicConstipation,
            @RequestParam(value="smoking",               defaultValue="false") boolean smoking,
            // Lifestyle
            @RequestParam(value="dietType",        required=false) String dietType,
            @RequestParam(value="activityLevel",   required=false) String activityLevel,
            @RequestParam(value="sleepQuality",    required=false) String sleepQuality,
            @RequestParam(value="stressLevel",     required=false) String stressLevel,
            // Clinical
            @RequestParam(value="hemoglobin",      required=false) Double hemoglobin,
            @RequestParam(value="bloodSugar",      required=false) Double bloodSugar,
            @RequestParam(value="crp",             required=false) Double crp,
            @RequestParam(value="cholesterol",     required=false) Double cholesterol,
            // Symptoms & colonoscopy
            @RequestParam(value="symptoms",        required=false) String symptoms,
            @RequestParam(value="polypLocation",   required=false) String polypLocation,
            @RequestParam(value="polypShape",      required=false) String polypShape,
            @RequestParam(value="surfaceTexture",  required=false) String surfaceTexture,
            @RequestParam(value="polypColor",      required=false) String polypColor) {

        String doctorId = extractDoctorId(authHeader);
        if (doctorId == null) return ResponseEntity.status(401).body(Map.of("message", "Unauthorized"));

        try {
            // Build entity
            PatientReport data = new PatientReport();
            data.setDoctorId(doctorId);
            data.setPatientId(patientId);
            data.setPatientName(patientName);
            data.setAge(age);
            data.setGender(gender);
            data.setHeight(height);
            data.setWeight(weight);
            data.setBloodGroup(bloodGroup);
            data.setEthnicity(ethnicity);
            data.setPrevColorectalSurgery(prevColorectalSurgery);
            data.setFamilyHistoryPolyps(familyHistoryPolyps);
            data.setIbd(ibd);
            data.setGeneticSyndrome(geneticSyndrome);
            data.setChronicConstipation(chronicConstipation);
            data.setSmoking(smoking);
            data.setDietType(dietType);
            data.setActivityLevel(activityLevel);
            data.setSleepQuality(sleepQuality);
            data.setStressLevel(stressLevel);
            data.setHemoglobin(hemoglobin);
            data.setBloodSugar(bloodSugar);
            data.setCrp(crp);
            data.setCholesterol(cholesterol);
            data.setSymptoms(symptoms);
            data.setPolypLocation(polypLocation);
            data.setPolypShape(polypShape);
            data.setSurfaceTexture(surfaceTexture);
            data.setPolypColor(polypColor);

            // Compute clinical risk score
            String riskScore = computeRiskScore(age, weight, height, smoking,
                    familyHistoryPolyps, ibd, geneticSyndrome, crp, bloodSugar);

            // Call Flask AI model
            Map<String, Object> flaskResult = flaskService.predict(image);
            String prediction = (String)  flaskResult.get("prediction");
            double confidence = ((Number) flaskResult.get("confidence")).doubleValue();
            String heatmap    = (String)  flaskResult.get("heatmap");

            PatientReport report = reportService.saveReport(
                    data, prediction, confidence, riskScore, heatmap, image);

            return ResponseEntity.ok(Map.of(
                    "id",         report.getId(),
                    "prediction", prediction,
                    "confidence", confidence,
                    "risk",       flaskResult.getOrDefault("risk", "Unknown"),
                    "riskScore",  riskScore,
                    "heatmap",    heatmap != null ? heatmap : "",
                    "bmi",        report.getBmi() != null ? report.getBmi() : 0
            ));
        } catch (Exception e) {
            return ResponseEntity.internalServerError()
                    .body(Map.of("message", "AI model error: " + e.getMessage()));
        }
    }

    @GetMapping("/reports/{doctorId}")
    public ResponseEntity<?> getReports(@RequestHeader("Authorization") String authHeader,
                                        @PathVariable String doctorId) {
        if (extractDoctorId(authHeader) == null)
            return ResponseEntity.status(401).body(Map.of("message", "Unauthorized"));
        return ResponseEntity.ok(reportService.getReportsByDoctorId(doctorId));
    }

    @GetMapping("/report/{id}")
    public ResponseEntity<?> getReport(@RequestHeader("Authorization") String authHeader,
                                       @PathVariable Long id) {
        if (extractDoctorId(authHeader) == null)
            return ResponseEntity.status(401).body(Map.of("message", "Unauthorized"));
        PatientReport report = reportService.getReportById(id);
        if (report == null) return ResponseEntity.notFound().build();
        return ResponseEntity.ok(report);
    }

    @GetMapping("/stats/{doctorId}")
    public ResponseEntity<?> getStats(@RequestHeader("Authorization") String authHeader,
                                      @PathVariable String doctorId) {
        if (extractDoctorId(authHeader) == null)
            return ResponseEntity.status(401).body(Map.of("message", "Unauthorized"));
        List<PatientReport> reports = reportService.getReportsByDoctorId(doctorId);
        long total   = reports.size();
        long polyp   = reports.stream().filter(r -> "Polyp Detected".equals(r.getPrediction())).count();
        long noPolyp = total - polyp;
        return ResponseEntity.ok(Map.of("total", total, "polyp", polyp, "noPolyp", noPolyp));
    }

    @GetMapping("/profile")
    public ResponseEntity<?> getProfile(@RequestHeader("Authorization") String authHeader) {
        String doctorId = extractDoctorId(authHeader);
        if (doctorId == null) return ResponseEntity.status(401).body(Map.of("message", "Unauthorized"));
        Doctor doctor = doctorService.findById(doctorId);
        if (doctor == null) return ResponseEntity.notFound().build();
        return ResponseEntity.ok(Map.of(
                "doctorId", doctor.getDoctorId(),
                "fullName", doctor.getFullName() != null ? doctor.getFullName() : "",
                "email",    doctor.getEmail()    != null ? doctor.getEmail()    : ""
        ));
    }

    // ── Multimodal Risk Score ─────────────────────────────────────────────────
    private String computeRiskScore(int age, Double weight, Double height,
                                    boolean smoking, boolean familyHistory,
                                    boolean ibd, boolean geneticSyndrome,
                                    Double crp, Double bloodSugar) {
        int score = 0;
        if (age >= 50)             score += 2;
        else if (age >= 40)        score += 1;
        if (smoking)               score += 2;
        if (familyHistory)         score += 3;
        if (ibd)                   score += 2;
        if (geneticSyndrome)       score += 3;
        if (crp != null && crp > 5)          score += 2;
        if (bloodSugar != null && bloodSugar > 126) score += 1;
        if (weight != null && height != null && height > 0) {
            double bmi = weight / Math.pow(height / 100.0, 2);
            if (bmi >= 30) score += 1;
        }
        if (score >= 7)      return "High";
        else if (score >= 4) return "Medium";
        else                 return "Low";
    }

    private String extractDoctorId(String authHeader) {
        if (authHeader == null || !authHeader.startsWith("Bearer ")) return null;
        try { return jwtUtil.extractDoctorId(authHeader.substring(7)); }
        catch (Exception e) { return null; }
    }
}
