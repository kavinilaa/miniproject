package com.medical.auth;

import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;

public interface PatientReportRepository extends JpaRepository<PatientReport, Long> {
    List<PatientReport> findByDoctorIdOrderByDateDesc(String doctorId);
}
