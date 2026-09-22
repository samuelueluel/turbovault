use std::io::{Cursor, Read};

use anyhow::{Context, Result, anyhow};
use quick_xml::Reader;
use quick_xml::events::{BytesStart, Event};
use zip::ZipArchive;

/// Extract a text-layer PDF as synthetic Markdown with one top-level section
/// per page. Page headings become stable retrieval locators after chunking.
pub(crate) fn extract_pdf_markdown(bytes: &[u8]) -> Result<String> {
    let pages = pdf_extract::extract_text_from_mem_by_pages(bytes)
        .map_err(|error| anyhow!("PDF text extraction failed: {error}"))?;
    let sections = pages
        .into_iter()
        .enumerate()
        .filter_map(|(index, text)| {
            let text = normalize_extracted_text(&text);
            (!text.is_empty()).then(|| format!("# Page {}\n\n{text}", index + 1))
        })
        .collect::<Vec<_>>();
    if sections.is_empty() {
        return Err(anyhow!(
            "PDF has no extractable text layer; OCR is required for scanned pages"
        ));
    }
    Ok(sections.join("\n\n"))
}

/// Extract paragraphs, headings, tabs, line breaks, and table-cell text from a
/// DOCX package. Word heading styles become Markdown headings so the common
/// hierarchical chunker can preserve the document's section structure.
pub(crate) fn extract_docx_markdown(bytes: &[u8]) -> Result<String> {
    let mut archive = ZipArchive::new(Cursor::new(bytes)).context("invalid DOCX ZIP container")?;
    let mut document = archive
        .by_name("word/document.xml")
        .context("DOCX has no word/document.xml")?;
    let mut xml = String::new();
    document
        .read_to_string(&mut xml)
        .context("DOCX document.xml is not UTF-8 XML")?;

    let mut reader = Reader::from_str(&xml);
    reader.config_mut().trim_text(false);
    let mut paragraphs = Vec::new();
    let mut paragraph = String::new();
    let mut paragraph_style = None;
    let mut in_paragraph = false;
    let mut in_text = false;

    loop {
        match reader.read_event().context("malformed DOCX document.xml")? {
            Event::Start(event) => match event.local_name().as_ref() {
                "p" => {
                    in_paragraph = true;
                    paragraph.clear();
                    paragraph_style = None;
                }
                "pStyle" if in_paragraph => {
                    paragraph_style = attribute_value(&event, "val");
                }
                "t" if in_paragraph => in_text = true,
                "tab" if in_paragraph => paragraph.push('\t'),
                "br" | "cr" if in_paragraph => paragraph.push('\n'),
                _ => {}
            },
            Event::Empty(event) => match event.local_name().as_ref() {
                "pStyle" if in_paragraph => {
                    paragraph_style = attribute_value(&event, "val");
                }
                "tab" if in_paragraph => paragraph.push('\t'),
                "br" | "cr" if in_paragraph => paragraph.push('\n'),
                _ => {}
            },
            Event::Text(text) if in_text => paragraph.push_str(text.as_ref()),
            Event::GeneralRef(reference) if in_text => {
                let decoded = match reference.as_ref() {
                    "amp" => "&",
                    "lt" => "<",
                    "gt" => ">",
                    "apos" => "'",
                    "quot" => "\"",
                    _ => "",
                };
                paragraph.push_str(decoded);
            }
            Event::End(event) => match event.local_name().as_ref() {
                "t" => in_text = false,
                "p" => {
                    in_paragraph = false;
                    in_text = false;
                    let text = paragraph.trim();
                    if !text.is_empty() {
                        if let Some(level) = paragraph_style.as_deref().and_then(docx_heading_level)
                        {
                            paragraphs.push(format!("{} {text}", "#".repeat(level)));
                        } else {
                            paragraphs.push(text.to_string());
                        }
                    }
                    paragraph.clear();
                    paragraph_style = None;
                }
                _ => {}
            },
            Event::Eof => break,
            _ => {}
        }
    }

    if paragraphs.is_empty() {
        return Err(anyhow!("DOCX has no extractable paragraph text"));
    }
    Ok(paragraphs.join("\n\n"))
}

fn attribute_value(event: &BytesStart<'_>, name: &str) -> Option<String> {
    event.attributes().flatten().find_map(|attribute| {
        let key = attribute.key.local_name();
        (key.as_ref() == name).then(|| attribute.value.as_ref().trim().to_string())
    })
}

fn docx_heading_level(style: &str) -> Option<usize> {
    let compact = style
        .chars()
        .filter(|character| !character.is_whitespace() && *character != '-')
        .flat_map(char::to_lowercase)
        .collect::<String>();
    compact
        .strip_prefix("heading")
        .and_then(|suffix| suffix.parse::<usize>().ok())
        .filter(|level| (1..=6).contains(level))
}

fn normalize_extracted_text(text: &str) -> String {
    text.replace("\r\n", "\n")
        .replace('\r', "\n")
        .lines()
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use zip::write::SimpleFileOptions;

    #[test]
    fn docx_headings_and_paragraphs_become_markdown() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Overview</w:t></w:r></w:p>
    <w:p><w:r><w:t>First &amp; second</w:t><w:tab/><w:t>column</w:t></w:r></w:p>
    <w:p><w:pPr><w:pStyle w:val="Heading 2"/></w:pPr><w:r><w:t>Details</w:t></w:r></w:p>
  </w:body>
</w:document>"#;
        let cursor = Cursor::new(Vec::new());
        let mut writer = zip::ZipWriter::new(cursor);
        writer
            .start_file("word/document.xml", SimpleFileOptions::default())
            .unwrap();
        writer.write_all(xml.as_bytes()).unwrap();
        let bytes = writer.finish().unwrap().into_inner();

        let markdown = extract_docx_markdown(&bytes).unwrap();
        assert_eq!(
            markdown,
            "# Overview\n\nFirst & second\tcolumn\n\n## Details"
        );
    }

    fn minimal_text_pdf() -> Vec<u8> {
        let stream = b"BT /F1 12 Tf 72 720 Td (Hello PDF attachment) Tj ET";
        let objects = [
            "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>".to_string(),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_string(),
            format!(
                "<< /Length {} >>\nstream\n{}\nendstream",
                stream.len(),
                String::from_utf8_lossy(stream)
            ),
        ];
        let mut pdf = b"%PDF-1.4\n".to_vec();
        let mut offsets = Vec::new();
        for (index, object) in objects.iter().enumerate() {
            offsets.push(pdf.len());
            pdf.extend_from_slice(format!("{} 0 obj\n{object}\nendobj\n", index + 1).as_bytes());
        }
        let xref = pdf.len();
        pdf.extend_from_slice(format!("xref\n0 {}\n", objects.len() + 1).as_bytes());
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        for offset in offsets {
            pdf.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
        }
        pdf.extend_from_slice(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
                objects.len() + 1
            )
            .as_bytes(),
        );
        pdf
    }

    #[test]
    fn text_layer_pdf_becomes_page_markdown() {
        let markdown = extract_pdf_markdown(&minimal_text_pdf()).unwrap();
        assert!(markdown.starts_with("# Page 1"));
        assert!(markdown.contains("Hello PDF attachment"));
    }

    #[test]
    fn scanned_or_empty_pdf_is_reported_as_unextractable() {
        assert!(extract_pdf_markdown(b"not a pdf").is_err());
    }

    #[test]
    fn recognizes_only_supported_word_heading_levels() {
        assert_eq!(docx_heading_level("Heading1"), Some(1));
        assert_eq!(docx_heading_level("heading 6"), Some(6));
        assert_eq!(docx_heading_level("Heading7"), None);
        assert_eq!(docx_heading_level("Title"), None);
    }
}
