<?php

namespace Tests\Feature;

use App\Models\IpcBatch;
use App\Models\IpcLog;
use App\Models\MasterProduct;
use App\Models\User;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Illuminate\Http\Client\Request;
use Illuminate\Http\UploadedFile;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Storage;
use Tests\TestCase;

class VisionAnalyzeTest extends TestCase
{
    use RefreshDatabase;

    protected function setUp(): void
    {
        parent::setUp();

        Storage::fake('public');
        config(['vision.base_url' => 'http://vision.test']);
    }

    private function makeBatch(?int $shelfLifeMonths = null, ?string $expDate = '2027-06-14'): IpcBatch
    {
        $product = MasterProduct::create([
            'fg_code' => 'FG-1',
            'product_name' => 'Product 1',
            'shelf_life_months' => $shelfLifeMonths,
            'is_active' => true,
        ]);

        return IpcBatch::create([
            'master_product_id' => $product->id,
            'no_batch' => 'BATCH-001',
            'exp_date' => $expDate,
            'created_by' => auth()->id() ?? User::factory()->create()->id,
            'current_stage' => IpcBatch::STAGE_PACKING,
        ]);
    }

    private function fakeVision(array $response): void
    {
        Http::fake(['vision.test/api/analyze' => Http::response($response)]);
    }

    private function visionResponse(array $overrides = []): array
    {
        return [
            'pipeline' => 'tube_emboss',
            'status' => 'OK',
            'extracted_date_code' => '140627',
            'raw_ocr_text' => 'EXP 140627',
            'confidence' => 0.97,
            'format_valid' => true,
            'engine_used' => 'paddleocr',
            'match_method' => 'label_anchor',
            'processing_time_ms' => 24123.4,
            ...$overrides,
        ];
    }

    private function analyze(IpcBatch $batch, string $fieldType)
    {
        return $this->postJson(route('vision.analyze', $batch), [
            'photo' => UploadedFile::fake()->image('tube.jpg'),
            'stage' => 'packing',
            'field_type' => $fieldType,
        ]);
    }

    public function test_confident_exp_matching_batch_exp_date_is_pass_and_logged(): void
    {
        $this->actingAs(User::factory()->create());
        $batch = $this->makeBatch();
        $this->fakeVision($this->visionResponse());

        $response = $this->analyze($batch, 'tube_exp_date');

        $response->assertOk()->assertJson([
            'decision' => 'PASS',
            'ocr_value' => '140627',
            'expected_value' => '140627',
        ]);

        $log = IpcLog::sole();
        $this->assertSame('OK', $log->vision_status);
        $this->assertSame(24123, $log->processing_time_ms);
        Storage::disk('public')->assertExists($log->image_path);

        Http::assertSent(function (Request $request) use ($log) {
            $fields = collect($request->data())->pluck('contents', 'name');

            return $request->url() === 'http://vision.test/api/analyze'
                && $fields['field_type'] === 'tube_exp_date'
                && $fields['request_id'] === $log->request_id
                && ! $fields->has('debug');
        });
    }

    public function test_confident_exp_not_matching_batch_is_fail(): void
    {
        $this->actingAs(User::factory()->create());
        $batch = $this->makeBatch(expDate: '2027-06-15');
        $this->fakeVision($this->visionResponse());

        $this->analyze($batch, 'tube_exp_date')->assertOk()->assertJson(['decision' => 'FAIL', 'expected_value' => '150627']);
    }

    public function test_low_confidence_is_review_even_when_value_matches(): void
    {
        $this->actingAs(User::factory()->create());
        $batch = $this->makeBatch();
        $this->fakeVision($this->visionResponse(['status' => 'LOW_CONFIDENCE', 'confidence' => 0.84]));

        $this->analyze($batch, 'tube_exp_date')->assertOk()->assertJson(['decision' => 'REVIEW']);
    }

    public function test_format_mismatch_is_review_and_keeps_full_ocr_text_out_of_ocr_value(): void
    {
        // Real failed-EXP response shape (2026-09-28, Pond's UV Protect): no extracted_date_code
        // key at all, and raw_ocr_text is every piece of text in the photo (460+ chars).
        $allText = 'EXP627MF00FZPOND\'SYRITYVUNIACINAMIDE-CMemberi perlindungan '.str_repeat('terhadap UVA/UVB ', 30);
        $response = $this->visionResponse([
            'status' => 'LOW_CONFIDENCE',
            'raw_ocr_text' => $allText,
            'confidence' => null,
            'format_valid' => false,
            'error_reason' => 'LABEL_NOT_FOUND',
        ]);
        unset($response['extracted_date_code']);

        $this->actingAs(User::factory()->create());
        $batch = $this->makeBatch();
        $this->fakeVision($response);

        $this->analyze($batch, 'tube_exp_date')->assertOk()->assertJson([
            'decision' => 'REVIEW',
            'ocr_value' => null,
            'error_reason' => 'LABEL_NOT_FOUND',
        ]);
        $this->assertSame($allText, IpcLog::sole()->raw_ocr_text);
    }

    public function test_missing_batch_exp_date_is_review(): void
    {
        $this->actingAs(User::factory()->create());
        $batch = $this->makeBatch(expDate: null);
        $this->fakeVision($this->visionResponse());

        $this->analyze($batch, 'tube_exp_date')->assertOk()->assertJson(['decision' => 'REVIEW']);
    }

    public function test_mfd_plus_shelf_life_matching_batch_exp_is_pass(): void
    {
        $this->actingAs(User::factory()->create());
        $batch = $this->makeBatch(shelfLifeMonths: 36);
        $this->fakeVision($this->visionResponse(['extracted_date_code' => '140624', 'raw_ocr_text' => 'MFD 140624 8 QFZ']));

        $this->analyze($batch, 'tube_mfd_date')->assertOk()->assertJson([
            'decision' => 'PASS',
            'ocr_value' => '140624',
            'computed_value' => '140627',
            'expected_value' => '140627',
        ]);
    }

    public function test_low_confidence_mfd_is_review_but_still_shows_computed_exp(): void
    {
        $this->actingAs(User::factory()->create());
        $batch = $this->makeBatch(shelfLifeMonths: 36);
        $this->fakeVision($this->visionResponse(['status' => 'LOW_CONFIDENCE', 'extracted_date_code' => '140624']));

        $this->analyze($batch, 'tube_mfd_date')->assertOk()->assertJson([
            'decision' => 'REVIEW',
            'computed_value' => '140627',
        ]);
    }

    public function test_mfd_without_shelf_life_on_product_is_review(): void
    {
        $this->actingAs(User::factory()->create());
        $batch = $this->makeBatch(shelfLifeMonths: null);
        $this->fakeVision($this->visionResponse(['extracted_date_code' => '140624']));

        $this->analyze($batch, 'tube_mfd_date')->assertOk()->assertJson(['decision' => 'REVIEW', 'computed_value' => null]);
    }

    public function test_field_type_without_a_rule_is_never_pass(): void
    {
        $this->actingAs(User::factory()->create());
        $batch = $this->makeBatch();
        $this->fakeVision($this->visionResponse(['extracted_date_code' => null, 'raw_ocr_text' => '310826QHB8']));

        $this->analyze($batch, 'tube_emboss_default')->assertOk()->assertJson(['decision' => 'REVIEW', 'ocr_value' => '310826QHB8']);
    }

    public function test_unreachable_vision_service_is_review_and_still_logged(): void
    {
        $this->actingAs(User::factory()->create());
        $batch = $this->makeBatch();
        Http::fake(['vision.test/*' => Http::failedConnection()]);

        $this->analyze($batch, 'tube_exp_date')->assertOk()->assertJson(['decision' => 'REVIEW', 'vision_status' => 'ERROR']);
        $this->assertStringStartsWith('VISION_UNREACHABLE', IpcLog::sole()->error_reason);
    }

    public function test_vision_decode_error_is_review(): void
    {
        $this->actingAs(User::factory()->create());
        $batch = $this->makeBatch();
        $this->fakeVision(['error' => 'Could not decode image. Unsupported or corrupt file.']);

        $this->analyze($batch, 'tube_exp_date')->assertOk()->assertJson(['decision' => 'REVIEW', 'vision_status' => 'ERROR']);
    }

    public function test_rejects_field_type_not_in_config(): void
    {
        $this->actingAs(User::factory()->create());
        $batch = $this->makeBatch();
        Http::fake();

        $this->analyze($batch, 'wo_number')->assertUnprocessable()->assertJsonValidationErrors('field_type');
        Http::assertNothingSent();
    }

    public function test_approver_cannot_analyze(): void
    {
        $this->actingAs(User::factory()->create(['role' => User::ROLE_APPROVER]));
        $batch = $this->makeBatch();
        Http::fake();

        $this->analyze($batch, 'tube_exp_date')->assertForbidden();
        Http::assertNothingSent();
    }

    public function test_logs_endpoint_lists_batch_history(): void
    {
        $this->actingAs(User::factory()->create());
        $batch = $this->makeBatch();
        $this->fakeVision($this->visionResponse());
        $this->analyze($batch, 'tube_exp_date');

        $this->getJson(route('vision.logs', $batch))->assertOk()->assertJsonCount(1)->assertJsonPath('0.decision', 'PASS');
    }
}
