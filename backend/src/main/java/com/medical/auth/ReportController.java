package com.medical.auth;

import com.itextpdf.text.*;
import com.itextpdf.text.pdf.*;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;

import java.io.ByteArrayOutputStream;
import java.time.format.DateTimeFormatter;
import java.util.Base64;

@RestController
@RequestMapping("/api/report")
@CrossOrigin(origins = "http://localhost:3000")
public class ReportController {

    private final PatientReportService reportService;
    private final DoctorService        doctorService;
    private final JwtUtil              jwtUtil;

    // Colours
    private static final BaseColor VIOLET     = new BaseColor(124, 58,  237);
    private static final BaseColor LIGHT_VIOLET = new BaseColor(192, 132, 252);
    private static final BaseColor DARK_BG    = new BaseColor(18,  18,  26);
    private static final BaseColor TEXT_LIGHT = new BaseColor(226, 232, 240);
    private static final BaseColor TEXT_MUTED = new BaseColor(107, 114, 128);
    private static final BaseColor RED        = new BaseColor(248, 113, 113);
    private static final BaseColor GREEN      = new BaseColor(74,  222, 128);

    public ReportController(PatientReportService reportService,
                            DoctorService doctorService,
                            JwtUtil jwtUtil) {
        this.reportService = reportService;
        this.doctorService = doctorService;
        this.jwtUtil       = jwtUtil;
    }

    // GET /api/report/pdf/{id}
    @GetMapping("/pdf/{id}")
    public ResponseEntity<byte[]> downloadPdf(
            @RequestHeader("Authorization") String authHeader,
            @PathVariable Long id) {

        if (!isAuthorized(authHeader))
            return ResponseEntity.status(401).build();

        PatientReport report = reportService.getReportById(id);
        if (report == null) return ResponseEntity.notFound().build();

        Doctor doctor = doctorService.findById(report.getDoctorId());

        try {
            byte[] pdf = buildPdf(report, doctor);
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_PDF);
            headers.setContentDisposition(ContentDisposition.attachment()
                    .filename("report_" + report.getPatientName().replace(" ", "_") + "_" + id + ".pdf")
                    .build());
            return ResponseEntity.ok().headers(headers).body(pdf);
        } catch (Exception e) {
            return ResponseEntity.internalServerError().build();
        }
    }

    private byte[] buildPdf(PatientReport report, Doctor doctor) throws Exception {
        Document doc = new Document(PageSize.A4, 50, 50, 60, 60);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PdfWriter writer = PdfWriter.getInstance(doc, out);
        doc.open();

        // Background
        PdfContentByte canvas = writer.getDirectContentUnder();
        canvas.setColorFill(DARK_BG);
        canvas.rectangle(0, 0, PageSize.A4.getWidth(), PageSize.A4.getHeight());
        canvas.fill();

        // Fonts
        Font titleFont   = new Font(Font.FontFamily.HELVETICA, 22, Font.BOLD,   LIGHT_VIOLET);
        Font sectionFont = new Font(Font.FontFamily.HELVETICA, 13, Font.BOLD,   LIGHT_VIOLET);
        Font labelFont   = new Font(Font.FontFamily.HELVETICA, 10, Font.BOLD,   new BaseColor(167, 139, 250));
        Font valueFont   = new Font(Font.FontFamily.HELVETICA, 10, Font.NORMAL, TEXT_LIGHT);
        Font mutedFont   = new Font(Font.FontFamily.HELVETICA,  9, Font.NORMAL, TEXT_MUTED);

        // ── Header ──
        Paragraph title = new Paragraph("🏥  MediCare Portal", titleFont);
        title.setAlignment(Element.ALIGN_CENTER);
        title.setSpacingAfter(4);
        doc.add(title);

        Paragraph sub = new Paragraph("Colorectal Cancer Detection — Scan Report", mutedFont);
        sub.setAlignment(Element.ALIGN_CENTER);
        sub.setSpacingAfter(20);
        doc.add(sub);

        addDivider(doc, writer);

        // ── Doctor Info ──
        doc.add(sectionParagraph("Doctor Information", sectionFont));
        doc.add(infoTable(new String[][]{
                {"Doctor ID",   report.getDoctorId()},
                {"Doctor Name", doctor != null && doctor.getFullName() != null ? doctor.getFullName() : "N/A"},
                {"Email",       doctor != null && doctor.getEmail()    != null ? doctor.getEmail()    : "N/A"},
                {"Report Date", report.getDate() != null
                        ? report.getDate().format(DateTimeFormatter.ofPattern("dd MMM yyyy, HH:mm"))
                        : "N/A"},
        }, labelFont, valueFont));

        doc.add(Chunk.NEWLINE);

        // ── Patient Info ──
        doc.add(sectionParagraph("Patient Information", sectionFont));
        doc.add(infoTable(new String[][]{
                {"Patient Name", report.getPatientName()},
                {"Age",          String.valueOf(report.getAge())},
                {"Gender",       report.getGender()},
                {"Symptoms",     report.getSymptoms() != null ? report.getSymptoms() : "N/A"},
        }, labelFont, valueFont));

        doc.add(Chunk.NEWLINE);

        // ── Prediction Result ──
        doc.add(sectionParagraph("AI Prediction Result", sectionFont));

        boolean isPolyp    = "Polyp Detected".equals(report.getPrediction());
        int     pct        = (int) Math.round(report.getConfidence() * 100);
        String  risk       = isPolyp ? (report.getConfidence() > 0.85 ? "High" : "Medium") : "Low";
        BaseColor riskColor = isPolyp ? RED : GREEN;

        doc.add(infoTable(new String[][]{
                {"Prediction",       report.getPrediction()},
                {"Confidence Score", pct + "%"},
                {"Risk Level",       risk},
                {"Condition",        "Colorectal Cancer Screening"},
        }, labelFont, valueFont));

        // Confidence bar
        doc.add(Chunk.NEWLINE);
        addConfidenceBar(doc, writer, pct, riskColor);

        // ── Heatmap ──
        if (report.getHeatmap() != null && !report.getHeatmap().isEmpty()) {
            doc.add(Chunk.NEWLINE);
            doc.add(sectionParagraph("Attention Heatmap (AI Focus Area)", sectionFont));
            try {
                byte[] imgBytes = Base64.getDecoder().decode(report.getHeatmap());
                Image heatmapImg = Image.getInstance(imgBytes);
                heatmapImg.scaleToFit(400, 220);
                heatmapImg.setAlignment(Element.ALIGN_CENTER);
                doc.add(heatmapImg);
                Paragraph heatLabel = new Paragraph("Model Focus Region — Grad-CAM Visualization", mutedFont);
                heatLabel.setAlignment(Element.ALIGN_CENTER);
                doc.add(heatLabel);
            } catch (Exception ignored) {}
        }

        // ── Footer ──
        doc.add(Chunk.NEWLINE);
        addDivider(doc, writer);
        Paragraph footer = new Paragraph(
                "Generated by MediCare Portal  •  Confidential Medical Document  •  " +
                new java.util.Date(), mutedFont);
        footer.setAlignment(Element.ALIGN_CENTER);
        doc.add(footer);

        doc.close();
        return out.toByteArray();
    }

    private Paragraph sectionParagraph(String text, Font font) {
        Paragraph p = new Paragraph(text, font);
        p.setSpacingBefore(10);
        p.setSpacingAfter(8);
        return p;
    }

    private PdfPTable infoTable(String[][] rows, Font labelFont, Font valueFont) throws DocumentException {
        PdfPTable table = new PdfPTable(2);
        table.setWidthPercentage(100);
        table.setWidths(new float[]{30f, 70f});
        table.setSpacingAfter(6);

        for (String[] row : rows) {
            PdfPCell labelCell = new PdfPCell(new Phrase(row[0], labelFont));
            labelCell.setBorder(Rectangle.BOTTOM);
            labelCell.setBorderColor(new BaseColor(30, 27, 75));
            labelCell.setBackgroundColor(new BaseColor(15, 15, 26));
            labelCell.setPadding(8);

            PdfPCell valueCell = new PdfPCell(new Phrase(row[1], valueFont));
            valueCell.setBorder(Rectangle.BOTTOM);
            valueCell.setBorderColor(new BaseColor(30, 27, 75));
            valueCell.setBackgroundColor(new BaseColor(18, 18, 26));
            valueCell.setPadding(8);

            table.addCell(labelCell);
            table.addCell(valueCell);
        }
        return table;
    }

    private void addDivider(Document doc, PdfWriter writer) throws DocumentException {
        PdfContentByte cb = writer.getDirectContent();
        cb.setColorStroke(VIOLET);
        cb.setLineWidth(0.5f);
        float y = writer.getVerticalPosition(true);
        cb.moveTo(50, y);
        cb.lineTo(PageSize.A4.getWidth() - 50, y);
        cb.stroke();
        doc.add(new Paragraph(" "));
    }

    private void addConfidenceBar(Document doc, PdfWriter writer, int pct, BaseColor color) throws DocumentException {
        float barWidth  = PageSize.A4.getWidth() - 100;
        float fillWidth = barWidth * pct / 100f;
        float y         = writer.getVerticalPosition(true) - 10;

        PdfContentByte cb = writer.getDirectContent();
        // Background track
        cb.setColorFill(new BaseColor(28, 28, 46));
        cb.roundRectangle(50, y, barWidth, 12, 6);
        cb.fill();
        // Fill
        cb.setColorFill(color);
        cb.roundRectangle(50, y, fillWidth, 12, 6);
        cb.fill();

        doc.add(new Paragraph("  "));
    }

    private boolean isAuthorized(String authHeader) {
        if (authHeader == null || !authHeader.startsWith("Bearer ")) return false;
        try { jwtUtil.extractDoctorId(authHeader.substring(7)); return true; }
        catch (Exception e) { return false; }
    }
}
