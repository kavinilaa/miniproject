package com.medical.auth;

import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Service;

@Service
public class DoctorService {

    private final DoctorRepository doctorRepository;
    private final BCryptPasswordEncoder passwordEncoder = new BCryptPasswordEncoder();

    public DoctorService(DoctorRepository doctorRepository) {
        this.doctorRepository = doctorRepository;
    }

    public boolean validate(String doctorId, String password) {
        return doctorRepository.findByDoctorId(doctorId)
                .map(doctor -> passwordEncoder.matches(password, doctor.getPassword()))
                .orElse(false);
    }

    public Doctor findById(String doctorId) {
        return doctorRepository.findByDoctorId(doctorId).orElse(null);
    }

    public String register(RegisterRequest request) {
        if (doctorRepository.existsByDoctorId(request.getDoctorId()))
            return "Doctor ID already exists.";
        if (doctorRepository.existsByEmail(request.getEmail()))
            return "Email already registered.";

        Doctor doctor = new Doctor();
        doctor.setDoctorId(request.getDoctorId());
        doctor.setFullName(request.getDoctorName());
        doctor.setEmail(request.getEmail());
        doctor.setPassword(passwordEncoder.encode(request.getPassword()));
        doctorRepository.save(doctor);
        return null;
    }
}
