package com.medical.auth;

public class LoginResponse {
    private String token;
    private String doctorId;
    private String fullName;
    private String email;
    private String message;

    public LoginResponse(String token, String doctorId, String fullName, String email, String message) {
        this.token    = token;
        this.doctorId = doctorId;
        this.fullName = fullName;
        this.email    = email;
        this.message  = message;
    }

    public String getToken()    { return token; }
    public String getDoctorId() { return doctorId; }
    public String getFullName() { return fullName; }
    public String getEmail()    { return email; }
    public String getMessage()  { return message; }
}
