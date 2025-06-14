import pdfplumber
import camelot
import logging
import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




def convert_df_to_rows(df: pd.DataFrame) -> list[str]:
    """Convert DataFrame to structured text rows"""
    if df.empty:
        return []
    
    # Clean column names
    df.columns = [str(col).strip() for col in df.columns]
    
    rows = []
    for _, rec in df.iterrows():
        cells = ["" if pd.isna(v) else str(v).strip() for v in rec]
        pairs = [f"{hdr}: {val}" for hdr, val in zip(df.columns, cells) if val and val != ""]
        if pairs:
            rows.append(" | ".join(pairs))
    return rows

def extract_tables_text(pdf_path: str) -> list[str]:
    """Extract tables from PDF using multiple methods"""
    rows = []
    
    # Method 1: Camelot lattice (for tables with clear borders)
    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
        for tbl in tables:
            if len(tbl.df) > 1:  # Skip single-row tables
                df = tbl.df.copy()
                if not df.iloc[0].isna().all():  # Use first row as header if not empty
                    df.columns = df.iloc[0]
                    df = df[1:].reset_index(drop=True)
                rows.extend(convert_df_to_rows(df))
    except Exception as e:
        logger.debug(f"Camelot lattice extraction failed: {str(e)}")
    
    # Method 2: pdfplumber (complementary approach)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table in tables:
                    if len(table) > 1:  # Skip single-row tables
                        df = pd.DataFrame(table[1:], columns=table[0])
                        table_rows = convert_df_to_rows(df)
                        # Add page reference to avoid duplicates
                        rows.extend([f"Page {page_num + 1} Table: {row}" for row in table_rows])
    except Exception as e:
        logger.debug(f"pdfplumber extraction failed: {str(e)}")
    
    return rows

rows = extract_tables_text("backend/shu_partitioned_db/pdfs/shu_academic_programs_db/approved-academic-calendar-m.phil_.-phd-program-spring-2025-.pdf")
for row in rows:
    print(row)