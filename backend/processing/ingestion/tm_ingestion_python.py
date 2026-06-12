"""Ingestion module for Transfermarkt using SeleniumBase (Cloudflare bypass)."""

import logging
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from seleniumbase import Driver  # type: ignore

logger = logging.getLogger(__name__)

BRONZE_DIR = Path("data/bronze")
BRONZE_DIR.mkdir(parents=True, exist_ok=True)


def extract_transfermarkt_squad_values(
    export_path: Path = BRONZE_DIR / "transfermarkt_raw.csv",
) -> pd.DataFrame:
    """Extract National Team squad values from Transfermarkt bypassing Cloudflare."""

    url = "https://www.transfermarkt.com/nationalmannschaften/marktwerte/statistik/nat/plus/0/galerie/0?kontinent_id=0"
    logger.info(f"Connecting to Transfermarkt via SeleniumBase: {url}")

    # Use Undetected Chromedriver to bypass Cloudflare/Cloudfront
    driver = Driver(uc=True, headless=True)

    try:
        driver.get(url)
        # Wait until the main table is loaded
        driver.uc_gui_click_captcha()  # Optional bypass if CF challenges
        driver.sleep(5)  # Wait for full render

        html = driver.get_page_source()
        soup = BeautifulSoup(html, "html.parser")

        # Parse the responsive table
        table = soup.find("table", {"class": "items"})
        if not table:
            raise ValueError("Could not find the 'items' table on Transfermarkt.")

        tbody = table.find("tbody")
        if not tbody:
            raise ValueError("Could not find tbody in the items table.")
        rows = tbody.find_all("tr")

        data = []
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 6:
                # Structure: Rank, Name, Logo, Squad Size, Average Age, Total Value
                # Name is usually in the second column as an anchor text
                nation = cols[1].text.strip()
                squad_size = cols[3].text.strip()
                total_value = cols[5].text.strip()

                data.append(
                    {"Nation": nation, "Squad": squad_size, "Total_Value": total_value}
                )

        df = pd.DataFrame(data)

        # Clean total value
        def clean_value(val_str: str) -> float:
            val = val_str.replace("€", "").strip()
            if "bn" in val:
                return float(val.replace("bn", "")) * 1e9
            elif "m" in val:
                return float(val.replace("m", "")) * 1e6
            elif "k" in val:
                return float(val.replace("k", "")) * 1e3
            return 0.0

        df["Total_Value_Num"] = df["Total_Value"].apply(clean_value)

        logger.info(
            f"Successfully scraped {len(df)} national teams from Transfermarkt."
        )

        df.to_csv(export_path, index=False)
        logger.info(f"Exported Transfermarkt Bronze data to {export_path}")

        return df

    except Exception as e:
        logger.error(f"Transfermarkt scraping failed: {e}")
        raise
    finally:
        driver.quit()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    extract_transfermarkt_squad_values()
