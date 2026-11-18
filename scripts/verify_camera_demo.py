"""Browser smoke against a running local demo; camera uses Chromium's fake device."""

import argparse
import json
import os
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8092")
parser.add_argument("--live-gemini", action="store_true")
parser.add_argument("--artifact-dir", type=Path)
parser.add_argument("--image", type=Path, help="Public bottle image for the real-detector smoke")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
image = args.image.resolve() if args.image else root / "data/local/smoke/bottle.jpg"
artifacts = args.artifact_dir.resolve() if args.artifact_dir else root / "artifacts"
(artifacts).mkdir(parents=True, exist_ok=True)
# Chromium subprocesses may write debug.log relative to cwd even with --log-file.
os.chdir(artifacts)
errors = []
checks = []
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(
        headless=True,
        args=[
            "--use-fake-ui-for-media-stream",
            "--use-fake-device-for-media-stream",
            "--log-file=" + str(artifacts / "chromium-debug.log"),
        ],
    )
    context = browser.new_context(viewport={"width": 1440, "height": 1100}, permissions=["camera"])
    page = context.new_page()
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(args.url, wait_until="networkidle")
    expect(page.locator("#capture")).to_be_enabled()
    page.locator("#capture").click()
    page.locator("#detect").wait_for(state="visible")
    assert page.locator("#camera").is_hidden()
    assert page.evaluate("document.querySelector('#camera').srcObject === null")
    checks.append("fake_camera_capture_stops_tracks_before_upload")
    previous_photo = page.locator("#photo").get_attribute("src")
    page.locator("#file").set_input_files(image)
    expect(page.locator("#photo")).not_to_have_attribute("src", previous_photo)
    expect(page.locator("#camera-status")).to_have_text("Tinjau foto")
    page.locator("#detect").click()
    page.locator(".bbox[aria-label^='Pilih Botol,']").wait_for(state="visible", timeout=25000)
    # Other hypotheses may cover the center; the outer border remains selectable.
    page.locator(".bbox[aria-label^='Pilih Botol,']").click(position={"x": 2, "y": 2})
    page.locator("#object-key").select_option("glass_bottle")
    page.locator("#cleanliness").select_option("CLEAN")
    page.locator("#integrity").select_option("INTACT")
    page.locator("#reuse-intention").select_option("false")
    page.locator("#recycling-access").select_option("YES")
    page.locator("#provider").select_option("fixture")
    page.locator("#confirmed").check()
    page.locator("#recommend").click()
    expect(page.locator("#method")).to_have_text("Recycle")
    assert page.locator("#recommendation-source").inner_text() == "Simulasi"
    checks.append("real_yolo_selectable_bbox_fixture_recommendation")
    page.screenshot(path=str(artifacts / "camera-demo-desktop.png"), full_page=True)
    # Changing context must discard a delayed answer without any provider request.
    page.evaluate("""() => {
      window.realDemoFetch = window.fetch;
      window.fetch = (url, options) => String(url).endsWith('/recommend')
        ? new Promise(resolve => setTimeout(() => resolve(new Response(JSON.stringify({data:{
            status:'READY',provider:'fixture',recommended_method:'RECYCLE',recommendation_scope:'HANDLING',
            reason:'STALE TEST RESPONSE',caution:'',steps:[],source_urls:[],guide_status:'DRAFT',evidence_ids:[]
          }}),{status:200,headers:{'Content-Type':'application/json'}})),500))
        : window.realDemoFetch(url,options);
    }""")
    page.locator("#recommend").click()
    page.locator("#cleanliness").select_option("DIRTY")
    page.wait_for_timeout(650)
    assert page.locator("#recommendation-result").is_hidden()
    assert "STALE TEST RESPONSE" not in page.locator("#reason").inner_text()
    page.evaluate("window.fetch = window.realDemoFetch")
    page.locator("#cleanliness").select_option("CLEAN")
    checks.append("late_recommendation_discarded_after_context_change")
    page.locator("#integrity").select_option("DAMAGED")
    assert page.locator("#recommendation-result").is_hidden()
    page.locator("#recommend").click()
    expect(page.locator("#method")).to_have_text("Trash")
    expect(page.locator("#caution")).to_contain_text("Perlu penanganan khusus")
    expect(page.locator("#source-list a")).to_have_attribute(
        "href", "https://www.cornwall.gov.uk/rubbish-recycling-and-waste/recycling/recycling-items/broken-glass/"
    )
    checks.append("broken_glass_keeps_trash_label_with_special_warning")
    if args.live_gemini:
        page.locator("#integrity").select_option("INTACT")
        page.locator("#provider").select_option("gemini")
        page.locator("#recommend").click()
        expect(page.locator("#recommendation-result")).to_be_visible(timeout=30000)
        assert page.locator("#method").inner_text() == "Recycle", page.locator("#global-message").inner_text()
        assert page.locator("#recommendation-source").inner_text() == "Gemini langsung"
        checks.append("live_gemini_browser_recommendation")
        page.screenshot(path=str(artifacts / "camera-demo-gemini.png"), full_page=True)
    page.locator("#integrity").select_option("INTACT")
    page.locator("#recommendation-goal").select_option("PREVENTION")
    page.locator("#recommend").click()
    expect(page.locator("#method")).to_have_text("Reduce", timeout=30000)
    expect(page.locator("#recommendation-scope")).to_contain_text("Pencegahan")
    expect(page.locator("#source-list a")).to_have_attribute(
        "href", "https://www.epa.gov/climate-change/what-you-can-do-about-climate-change-waste"
    )
    assert page.locator("#sources").get_attribute("open") is not None
    assert page.locator("#alternative").count() == 0
    checks.append("reduce_is_single_prevention_result_with_matching_source")
    page.locator("#catalog-search").fill("Buku kertas")
    page.locator(".catalog-item").click()
    page.locator("#cleanliness").select_option("CLEAN")
    page.locator("#integrity").select_option("INTACT")
    page.locator("#reusable").select_option("YES")
    page.locator("#reuse-intention").select_option("true")
    page.locator("#confirmed").check()
    page.locator("#recommend").click()
    expect(page.locator("#method")).to_have_text("Reuse", timeout=30000)
    assert page.locator("#reason").inner_text() and page.locator("#source-list a").count()
    checks.append("reuse_book_with_confirmed_condition_reason_and_sources")
    page.locator("#catalog-search").fill("Sikat gigi plastik manual")
    page.locator(".catalog-item").click()
    page.locator("#reusable").select_option("NO")
    page.locator("#recycling-access").select_option("NO")
    page.locator("#confirmed").check()
    page.locator("#recommend").click()
    expect(page.locator("#method")).to_have_text("Trash", timeout=30000)
    expect(page.locator("#source-list a")).to_have_attribute(
        "href", "https://www.cornwall.gov.uk/rubbish-recycling-and-waste/recycling/recycling-items/toothbrushes/"
    )
    checks.append("trash_manual_plastic_toothbrush_with_confirmed_no_recycling")
    page.locator("#catalog-search").fill("baterai")
    page.locator(".catalog-item").filter(has=page.get_by_text("Baterai", exact=True)).click()
    assert page.locator("#selected-label").inner_text() == "Pilihan manual"
    page.locator("#recycling-access").select_option("YES")
    page.locator("#confirmed").check()
    page.locator("#recommend").click()
    expect(page.locator("#method")).to_have_text("Recycle", timeout=30000)
    expect(page.locator("#caution")).to_contain_text("Perlu penanganan khusus")
    expect(page.locator("#handling-steps")).to_contain_text("layanan khusus")
    checks.append("battery_keeps_recycle_label_and_specialist_collection_warning")
    page.locator("#catalog-search").fill("Gelas kaca")
    page.locator(".catalog-item").click()
    page.locator("#integrity").select_option("DAMAGED")
    page.locator("#confirmed").check()
    page.locator("#recommend").click()
    expect(page.locator("#method")).to_have_text("Trash", timeout=30000)
    expect(page.locator("#caution")).to_contain_text("Perlu penanganan khusus")
    expect(page.locator("#handling-steps")).not_to_contain_text("Gunakan kembali hanya")
    expect(page.locator("#source-list a")).to_have_attribute(
        "href", "https://www.cornwall.gov.uk/rubbish-recycling-and-waste/recycling/recycling-items/broken-glass/"
    )
    if args.live_gemini:
        expect(page.locator("#recommendation-source")).to_have_text("Gemini langsung")
    checks.append("broken_drinking_glass_has_trash_reason_sources_and_no_reuse_steps")
    page.screenshot(path=str(artifacts / "camera-demo-special-classification.png"), full_page=True)
    page.locator("#catalog-search").fill("Cangkir keramik")
    page.locator(".catalog-item").click()
    page.locator("#cleanliness").select_option("DIRTY")
    expect(page.locator("#cleaning-details")).to_be_visible()
    page.locator("#integrity").select_option("INTACT")
    page.locator("#reusable").select_option("YES")
    page.locator("#reuse-intention").select_option("true")
    page.locator("#soiling").select_option("FOOD_DRINK")
    page.locator("#cleanable").select_option("YES")
    page.locator("#risk").check()
    expect(page.locator("#risk-details")).to_be_visible()
    page.locator("#confirmed").check()
    page.locator("#recommend").click()
    expect(page.locator("#method")).to_have_text("Lengkapi kondisi · panduan awal")
    expect(page.locator("#reason")).to_contain_text("jenisnya belum dijelaskan")
    checks.append("unknown_risk_asks_for_context_instead_of_reduce")
    page.locator("#risk").uncheck()
    page.locator("#recommend").click()
    expect(page.locator("#method")).to_have_text("Reuse", timeout=30000)
    expect(page.locator("#reason")).to_contain_text("setelah dicuci")
    expect(page.locator("#handling-steps")).to_contain_text("cuci")
    assert page.locator("#cleanliness").input_value() == "DIRTY"
    expect(page.locator("#source-list")).to_contain_text("Pemerintah Kabupaten Banjarnegara")
    expect(page.locator("#source-list")).to_contain_text("Amerika Serikat")
    checks.append("washable_dirty_ceramic_mug_is_reuse_with_scoped_sources")
    page.screenshot(path=str(artifacts / "camera-demo-washable-mug.png"), full_page=True)
    page.locator("#soiling").select_option("HAZARDOUS")
    expect(page.locator("#risk")).to_be_checked()
    page.locator("#recommend").click()
    expect(page.locator("#method")).to_have_text("Panduan penanganan khusus")
    expect(page.locator("#reason-code")).to_contain_text("penanganan khusus")
    assert "Reduce" not in page.locator("#method").inner_text()
    checks.append("hazardous_contamination_does_not_get_ordinary_washing_or_reduce")
    page.locator("#catalog-search").fill("Cangkir keramik")
    page.locator(".catalog-item").click()
    page.locator("#provider").select_option("fixture")
    page.locator("#confirmed").check()
    page.locator("#recommend").click()
    expect(page.locator("#method")).to_have_text("Reuse", timeout=30000)
    expect(page.locator("#recommendation-basis")).to_contain_text("Saran sementara")
    expect(page.locator("#reason")).to_contain_text("belum lengkap")
    assert page.locator("#cleanliness").input_value() == "UNKNOWN"
    assert page.locator("#integrity").input_value() == "UNKNOWN"
    assert page.locator("#reuse-intention").input_value() == ""
    page.screenshot(path=str(artifacts / "camera-demo-conditional.png"), full_page=True)
    checks.append("empty_optional_context_returns_conditional_advice_without_filling_unknowns")
    page.locator("#reusable").select_option("NO")
    expect(page.locator("#recommendation-result")).to_be_hidden()
    checks.append("conditional_advice_cleared_when_known_condition_changes")
    for label in ("Kaleng baja/tinplate", "Wadah makanan plastik / foam"):
        page.locator("#catalog-search").fill(label)
        page.locator(".catalog-item").click()
        expect(page.locator("#former-contents-field")).to_be_visible()
        assert page.locator("#former-contents").input_value() == "UNKNOWN"
        page.locator("#cleanliness").select_option("CLEAN")
        page.locator("#recycling-access").select_option("YES")
        page.locator("#confirmed").check()
        page.locator("#recommend").click()
        expect(page.locator("#method")).to_have_text("Lengkapi kondisi · panduan awal")
        expect(page.locator("#reason-code")).to_contain_text("isi sebelumnya")
        page.locator("#former-contents").select_option("FOOD_DRINK")
        expect(page.locator("#recommendation-result")).to_be_hidden()
        page.locator("#recommend").click()
        expect(page.locator("#method")).to_have_text("Recycle")
        page.locator("#former-contents").select_option("CHEMICAL")
        page.locator("#recommend").click()
        expect(page.locator("#method")).to_have_text("Panduan penanganan khusus")
        checks.append("prior_contents_reachable_and_hazard_blocked_" + label)
    page.locator("#catalog-search").fill("Botol kaca")
    page.locator(".catalog-item").click()
    page.locator("#provider").select_option("fixture")
    page.locator("#cleanliness").select_option("CLEAN")
    page.locator("#integrity").select_option("INTACT")
    page.locator("#reusable").select_option("YES")
    page.locator("#reuse-intention").select_option("true")
    page.locator("#recycling-access").select_option("YES")
    page.locator("#risk").check()
    page.locator("#risk-type").select_option("SHARP")
    page.locator("#confirmed").check()
    page.locator("#recommend").click()
    expect(page.locator("#method")).to_have_text("Recycle")
    expect(page.locator("#reason")).to_contain_text("kondisi tajam")
    expect(page.locator("#caution")).to_contain_text("Perlu penanganan khusus")
    assert page.locator("#handling-steps li").count() >= 3
    page.screenshot(path=str(artifacts / "policy08-sharp-intact.png"), full_page=True)
    checks.append("sharp_intact_glass_has_recycle_method_and_warning")
    page.locator("#integrity").select_option("DAMAGED")
    page.locator("#recommend").click()
    expect(page.locator("#method")).to_have_text("Trash")
    checks.append("confirming_broken_glass_still_returns_supported_trash_method")
    for label, heading in (
        ("Barang lain / bahan belum diketahui", "Recycle"),
        ("Elektronik / baterai lain", "Recycle"),
    ):
        page.locator("#catalog-search").fill(label)
        page.locator(".catalog-item").click()
        page.locator("#provider").select_option("fixture")
        page.locator("#confirmed").check()
        page.locator("#recommend").click()
        expect(page.locator("#method")).to_have_text(heading)
        assert page.locator("#handling-steps li").count() >= 3
        assert page.locator("#source-list a").count() >= 1
        checks.append("uncatalogued_item_has_conditional_method_" + label)
    page.screenshot(path=str(artifacts / "policy08-other-electronics.png"), full_page=True)
    expect(page.locator("#model-coverage")).to_contain_text("Model aktif:")
    page.locator(".object-choice:not(:disabled)").first.click()
    expect(page.locator("#correct-detection")).to_be_visible()
    page.locator("#object-key").select_option("cardboard")
    page.locator("#correct-detection").click()
    expect(page.locator("#selected-label")).to_have_text("Pilihan manual")
    page.locator("#provider").select_option("fixture")
    page.locator("#cleanliness").select_option("CLEAN")
    page.locator("#recycling-access").select_option("YES")
    page.locator("#confirmed").check()
    with page.expect_response(lambda r: r.url.endswith('/demo/api/recommend')) as response:
        page.locator("#recommend").click()
    assert response.value.json()["data"]["input_source"] == "manual_confirmation"
    expect(page.locator("#method")).to_have_text("Recycle")
    checks.append("explicit_wrong_detection_correction_uses_manual_cardboard_context")
    page.evaluate("""async () => {
      const nonce = document.querySelector('meta[name="wastenot-demo"]').content;
      for (let i=0; i<31; i++) {
        const response = await fetch('/demo/api/recommend', {method:'POST',headers:{'Content-Type':'application/json','X-WasteNot-Demo':nonce},body:JSON.stringify({client_request_id:crypto.randomUUID(),manual:true,confirmed_object:true,provider:'fixture',context:{object_key:'pet_bottle',category:'PLASTIC',condition:{cleanliness:'CLEAN'},recycling_access:'YES'}})});
        if (!response.ok) throw new Error('Recommendation failed: '+response.status);
        if ((await response.json()).data.provider === 'rules') break;
      }
    }""")
    page.locator("#cleanliness").select_option("DIRTY")
    page.locator("#cleanliness").select_option("CLEAN")
    page.locator("#recommend").click()
    expect(page.locator("#method")).to_have_text("Recycle")
    expect(page.locator("#recommendation-source")).to_have_text("Aturan panduan")
    expect(page.locator("#reason-code")).to_contain_text("Saran tetap tersedia")
    assert page.locator("#handling-steps li").count() >= 3
    checks.append("generation_budget_keeps_sourced_rule_method")
    page.locator("#catalog-search").fill("Barang lain / bahan belum diketahui")
    page.locator(".catalog-item").click()
    page.locator("#provider").select_option("fixture")
    page.locator("#confirmed").check()
    page.locator("#recommend").click()
    expect(page.locator("#method")).to_have_text("Recycle")
    assert page.locator("#handling-steps li").count() >= 3
    checks.append("generic_item_keeps_conditional_method_after_generation_budget")
    page.screenshot(path=str(artifacts / "policy09-method-after-limit.png"), full_page=True)
    page.set_viewport_size({"width": 390, "height": 844})
    page.locator("#clear").click()
    expect(page.locator("#photo-stage")).to_be_hidden()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    page.screenshot(path=str(artifacts / "camera-demo-mobile.png"), full_page=True)
    checks.append("mobile_layout_and_photo_cleanup")
    # Denial is exercised with a fresh browser context and a controlled API rejection.
    denied = browser.new_context(viewport={"width": 390, "height": 844})
    denied.add_init_script(
        "Object.defineProperty(navigator.mediaDevices,'getUserMedia',{value:()=>Promise.reject(new DOMException('Denied','NotAllowedError'))})"
    )
    denied_page = denied.new_page()
    denied_page.goto(args.url, wait_until="networkidle")
    expect(denied_page.locator("#global-message")).to_contain_text("Izin kamera")
    assert denied_page.locator("#choose-file").is_visible() and denied_page.locator("#manual").is_visible()
    checks.append("camera_denied_keeps_file_and_manual_fallback")
    assert not errors, errors
    report = {
        "checks": checks,
        "browser_errors": errors,
        "camera": "fake Chromium device; physical camera not accessed",
        "real_yolo": True,
        "live_gemini": args.live_gemini,
    }
    (artifacts / "camera-demo-browser.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    browser.close()
