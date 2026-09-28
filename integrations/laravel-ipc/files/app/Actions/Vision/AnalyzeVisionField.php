<?php

namespace App\Actions\Vision;

use App\Models\IpcBatch;
use App\Models\IpcLog;
use App\Models\User;
use App\Services\Vision\VisionClient;
use App\Services\Vision\VisionEvaluator;
use Illuminate\Http\UploadedFile;
use Illuminate\Support\Str;

/**
 * Stores the photo, sends it to the Vision Service, lets VisionEvaluator decide
 * PASS / FAIL / REVIEW, and writes one ipc_logs row. The photo is kept under its own
 * "vision" folder and NOT as an IpcAttachment, so it never changes which photo a stage's
 * edit page / printed report treats as the current one for a field.
 */
class AnalyzeVisionField
{
    public function __construct(
        private readonly VisionClient $client,
        private readonly VisionEvaluator $evaluator,
    ) {}

    public function handle(IpcBatch $batch, User $user, UploadedFile $photo, string $stage, string $fieldType): IpcLog
    {
        $requestId = (string) Str::uuid();
        $path = $photo->storeAs("ipc-attachments/{$batch->id}/vision", "{$requestId}.{$photo->extension()}", 'public');

        $result = $this->client->analyze($photo->get(), $photo->getClientOriginalName() ?: 'capture.jpg', $fieldType, $requestId);
        $evaluation = $this->evaluator->evaluate($batch->loadMissing('masterProduct'), $fieldType, $result);

        return IpcLog::create([
            'request_id' => $requestId,
            'ipc_batch_id' => $batch->id,
            'stage' => $stage,
            'field_type' => $fieldType,
            'image_path' => $path,
            'vision_status' => $result->status,
            'engine_used' => $result->engineUsed,
            'confidence' => $result->confidence,
            'format_valid' => $result->formatValid,
            'ocr_value' => $result->value,
            'raw_ocr_text' => $result->rawText,
            'error_reason' => $result->errorReason,
            'match_method' => $result->matchMethod,
            'processing_time_ms' => $result->processingTimeMs,
            'vision_response' => $result->raw ?: null,
            'decision' => $evaluation->decision,
            'decision_reason' => $evaluation->reason,
            'expected_value' => $evaluation->expectedValue,
            'computed_value' => $evaluation->computedValue,
            'created_by' => $user->id,
        ]);
    }
}
