import sys
import os
import time

sys.stdout.reconfigure(encoding='utf-8')

from playwright.sync_api import sync_playwright

def run_rpa_test():
    print("=" * 60)
    print("🤖 EJECUTANDO PRUEBA RPA DE INTERFAZ GRÁFICA EN PLAYWRIGHT")
    print("=" * 60)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        print("🌐 Navegando a http://localhost:8501 ...")
        page.goto("http://localhost:8501", timeout=30000)
        page.wait_for_load_state("networkidle")
        
        print("✅ Página cargada. Buscando cuadro de entrada de chat (st.chat_input)...")
        chat_input = page.locator("textarea[data-testid='stChatInputTextArea']")
        chat_input.wait_for(state="visible", timeout=10000)
        
        test_prompt = "¿Qué sanciones se establecen en el Artículo 62?"
        print(f"📝 Escribiendo consulta del usuario: '{test_prompt}' ...")
        chat_input.fill(test_prompt)
        time.sleep(1)
        
        print("⌨️ Presionando Enter para enviar el mensaje...")
        chat_input.press("Enter")
        
        print("⏳ Esperando respuesta del asistente RAG y barra de progreso (spinner)...")
        page.wait_for_selector(".stChatMessage[data-testid='stChatMessageAssistant']", timeout=25000)
        
        time.sleep(3)
        
        # Tomar captura de pantalla de verificación RPA
        screenshot_path = "rpa_chat_test_screenshot.png"
        page.screenshot(path=screenshot_path, full_page=True)
        print(f"📸 Captura de pantalla RPA guardada exitosamente como '{screenshot_path}'!")
        
        # Extraer mensajes visibles
        messages = page.locator(".stChatMessage").all_text_contents()
        print(f"💬 Total de burbujas de chat renderizadas en pantalla: {len(messages)}")
        for idx, msg in enumerate(messages, 1):
            print(f"--- Mensaje #{idx} ---\n{msg[:150]}...\n")
            
        browser.close()
        
    print("=" * 60)
    print("🎉 PRUEBA RPA FINALIZADA CON ÉXITO")
    print("=" * 60)

if __name__ == "__main__":
    run_rpa_test()
