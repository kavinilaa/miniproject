package com.medical.auth;

import org.springframework.data.jpa.repository.JpaRepository;
import java.util.Optional;

public interface DoctorRepository extends JpaRepository<Doctor, String> {
    Optional<Doctor> findByDoctorId(String doctorId);
    boolean existsByDoctorId(String doctorId);
    boolean existsByEmail(String email);
}
