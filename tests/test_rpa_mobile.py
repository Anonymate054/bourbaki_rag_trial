import sys
import os
import time
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding='utf-8')

def test_mobile_responsive():
    print("=" * 60)
    print("📱 PRUEBA DE RESPONSIVIDAD Y NAVEGACIÓN EN DISPOSITIVO MÓVIL (PLAYWRIGHT)")
    print("=" * 60)
    
    with sync_playwright() as p:
        # Emulate Mobile Phone Viewport (iPhone 12: 390x844)
        mobile_device = p.devices['iPhone 12']
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(**mobile_device)
        page = context.new_page()
        
        print("🌐 1. Navegando a http://localhost:8501 en modo MÓVIL (390x844)...")
        page.goto("http://localhost:8501", timeout=45000)
        time.sleep(4)
        
        # Take Mobile Screenshot of Main Screen (Sidebar Collapsed)
        page.screenshot(path="rpa_mobile_collapsed.png", full_page=False)
        print("📸 2. Captura de pantalla móvil (Barra Lateral Plegada): 'rpa_mobile_collapsed.png'!")
        
        # Locate Streamlit Sidebar Toggle Button (Chevron / Hamburger)
        sidebar_toggle = page.locator("[data-testid='stSidebarCollapseButton'], button[aria-label='Open sidebar']")
        if sidebar_toggle.count() > 0:
            print("👉 3. Haciendo clic en el botón de despliegue de barra lateral en móvil...")
            sidebar_toggle.first.click()
            time.sleep(2)
            
            page.screenshot(path="rpa_mobile_expanded.png", full_page=False)
            print("📸 4. Captura de pantalla móvil (Barra Lateral Desplegada): 'rpa_mobile_expanded.png'!")
        else:
            print("ℹ️ Botón de toggle no encontrado directamente, verificando estado...")
            
        browser.close()
        
    print("=" * 60)

if __name__ == "__main__":
    test_mobile_responsive()
