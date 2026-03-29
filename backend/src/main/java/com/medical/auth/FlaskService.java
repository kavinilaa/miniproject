package com.medical.auth;

import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.util.Map;

@Service
public class FlaskService {

    private static final String FLASK_URL = "http://localhost:5000/predict";
    private final RestTemplate restTemplate = new RestTemplate();

    @SuppressWarnings("unchecked")
    public Map<String, Object> predict(MultipartFile imageFile) throws Exception {
        byte[] bytes = imageFile.getBytes();

        ByteArrayResource resource = new ByteArrayResource(bytes) {
            @Override public String getFilename() { return imageFile.getOriginalFilename(); }
        };

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("image", resource);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        HttpEntity<MultiValueMap<String, Object>> request = new HttpEntity<>(body, headers);
        ResponseEntity<Map> response = restTemplate.postForEntity(FLASK_URL, request, Map.class);

        if (response.getStatusCode() != HttpStatus.OK || response.getBody() == null)
            throw new RuntimeException("Flask model returned error");

        return response.getBody();
    }
}
