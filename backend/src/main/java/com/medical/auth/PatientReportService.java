package com.medical.auth;

import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.*;
import java.util.List;
import java.util.UUID;

@Service
public class PatientReportService {

    private final PatientReportRepository reportRepository;
    private static final String UPLOAD_DIR = "uploads/";

    public PatientReportService(PatientReportRepository reportRepository) {
        this.reportRepository = reportRepository;
    }

    public PatientReport saveReport(PatientReport data,
                                    String prediction, double confidence,
                                    String riskScore, String heatmap,
                                    MultipartFile imageFile) throws IOException {
        // Auto-calculate BMI
        if (data.getHeight() != null && data.getWeight() != null && data.getHeight() > 0) {
            double heightM = data.getHeight() / 100.0;
            data.setBmi(Math.round((data.getWeight() / (heightM * heightM)) * 10.0) / 10.0);
        }
        data.setPrediction(prediction);
        data.setConfidence(confidence);
        data.setRiskScore(riskScore);
        data.setHeatmap(heatmap);
        data.setImagePath(saveImage(imageFile));
        return reportRepository.save(data);
    }

    public List<PatientReport> getReportsByDoctorId(String doctorId) {
        return reportRepository.findByDoctorIdOrderByDateDesc(doctorId);
    }

    public PatientReport getReportById(Long id) {
        return reportRepository.findById(id).orElse(null);
    }

    private String saveImage(MultipartFile file) throws IOException {
        if (file == null || file.isEmpty()) return null;
        Path uploadPath = Paths.get(UPLOAD_DIR);
        if (!Files.exists(uploadPath)) Files.createDirectories(uploadPath);
        String filename = UUID.randomUUID() + "_" + file.getOriginalFilename();
        Files.copy(file.getInputStream(), uploadPath.resolve(filename), StandardCopyOption.REPLACE_EXISTING);
        return UPLOAD_DIR + filename;
    }
}
