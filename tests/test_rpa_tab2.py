import sys
import os
import time
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding='utf-8')

def test_tab2():
    print("=" * 60)
    print("🤖 PRUEBA Y DIAGNÓSTICO ESPECÍFICO DE LA PESTAÑA 2 (VISUALIZADOR DE BD)")
    print("=" * 60)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        print("🌐 1. Navegando a http://localhost:8501 ...")
        page.goto("http://localhost:8501", timeout=45000)
        time.sleep(4)
        
        # Locate tabs with role="tab" or [data-testid="stTab"]
        tabs = page.locator("[data-testid='stTab'], button[role='tab']")
        print(f"Encontradas {tabs.count()} pestañas en pantalla:")
        for idx in range(tabs.count()):
            print(f" - Pestaña #{idx+1}: {tabs.nth(idx).inner_text()}")
            
        if tabs.count() >= 2:
            print("👉 Haciendo clic en la Pestaña 2...")
            tabs.nth(1).click()
            time.sleep(3)
            
            page.screenshot(path="rpa_tab2_screenshot.png", full_page=True)
            print("📸 Captura de pantalla de la Pestaña 2 guardada como 'rpa_tab2_screenshot.png'!")
            
            # Check Streamlit Alerts or Exception Tracebacks
            alerts = page.locator("[data-testid='stAlert'], .element-container:has-text('Traceback'), .stException").all_text_contents()
            if alerts:
                print(f"\n⚠️ Alertas / Excepciones encontradas en Pestaña 2 ({len(alerts)}):")
                for a in alerts:
                    print(f"  - {a.strip()}\n")
            else:
                print("✅ No se detectaron excepciones en la Pestaña 2.")
                
            headings = page.locator("h1, h2, h3, h4, h5").all_text_contents()
            print("\nEncabezados Renderizados en Pestaña 2:")
            for h in headings:
                print(f" - {h.strip()}")
        else:
            print("❌ Menos de 2 pestañas encontradas.")
            
        browser.close()
        
    print("=" * 60)

if __name__ == "__main__":
    test_tab2()
