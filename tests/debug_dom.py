import sys
import os
import time
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding='utf-8')

def debug_dom():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("http://localhost:8501", timeout=30000)
        page.wait_for_load_state("networkidle")
        time.sleep(2)
        
        # Get all elements with data-testid attribute
        test_ids = page.eval_on_selector_all("[data-testid]", "elements => elements.map(e => e.getAttribute('data-testid'))")
        print("DOM data-testid attributes found:", set(test_ids))
        
        # Get all elements with stChatMessage in class
        chat_classes = page.eval_on_selector_all(".stChatMessage, div[class*='stChatMessage']", "elements => elements.map(e => e.className + ' | ' + e.innerText)")
        print(f"\nFound {len(chat_classes)} chat elements:")
        for c in chat_classes[:5]:
            print(f" - {c[:150]}")
            
        browser.close()

if __name__ == "__main__":
    debug_dom()
