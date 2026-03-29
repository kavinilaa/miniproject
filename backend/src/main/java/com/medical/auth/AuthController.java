package com.medical.auth;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;
import java.util.regex.Pattern;

@RestController
@RequestMapping("/api/auth")
@CrossOrigin(origins = "http://localhost:3000")
public class AuthController {

    private static final Pattern EMAIL_PATTERN =
            Pattern.compile("^[A-Za-z0-9+_.-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$");

    private final DoctorService doctorService;
    private final JwtUtil jwtUtil;

    public AuthController(DoctorService doctorService, JwtUtil jwtUtil) {
        this.doctorService = doctorService;
        this.jwtUtil = jwtUtil;
    }

    @PostMapping("/login")
    public ResponseEntity<?> login(@RequestBody LoginRequest request) {
        if (isBlank(request.getDoctorId()) || isBlank(request.getPassword()))
            return bad("All fields are required.");

        if (!doctorService.validate(request.getDoctorId(), request.getPassword()))
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(msg("Invalid Doctor ID or Password."));

        Doctor doctor = doctorService.findById(request.getDoctorId());
        String token = jwtUtil.generateToken(request.getDoctorId());
        return ResponseEntity.ok(new LoginResponse(
                token,
                doctor.getDoctorId(),
                doctor.getFullName() != null ? doctor.getFullName() : "",
                doctor.getEmail()    != null ? doctor.getEmail()    : "",
                "Login successful."
        ));
    }

    @PostMapping("/register")
    public ResponseEntity<?> register(@RequestBody RegisterRequest request) {
        if (isBlank(request.getDoctorName()) || isBlank(request.getDoctorId()) ||
            isBlank(request.getEmail()) || isBlank(request.getPassword()))
            return bad("All fields are required.");

        if (!EMAIL_PATTERN.matcher(request.getEmail()).matches())
            return bad("Invalid email format.");

        String error = doctorService.register(request);
        if (error != null)
            return ResponseEntity.status(HttpStatus.CONFLICT).body(msg(error));

        return ResponseEntity.status(HttpStatus.CREATED).body(msg("Doctor registered successfully."));
    }

    private boolean isBlank(String value) {
        return value == null || value.isBlank();
    }

    private Map<String, String> msg(String message) {
        return Map.of("message", message);
    }

    private ResponseEntity<?> bad(String message) {
        return ResponseEntity.badRequest().body(msg(message));
    }
}
