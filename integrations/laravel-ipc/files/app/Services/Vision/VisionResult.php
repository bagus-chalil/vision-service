<?php

namespace App\Services\Vision;

/**
 * What the Vision Service read — never a decision. `value` is the OCR'd code as-is, never
 * trimmed or auto-corrected (same rule as the Vision Service itself): extracted_date_code for
 * EXP/MFD fields, or raw_ocr_text for tube_emboss_default — the latter only when format_valid,
 * because on a failed EXP/MFD search raw_ocr_text is every piece of text in the photo (400+
 * chars), not a code. `rawText` always keeps raw_ocr_text for the audit log.
 */
class VisionResult
{
    public const STATUS_OK = 'OK';

    public const STATUS_LOW_CONFIDENCE = 'LOW_CONFIDENCE';

    public const STATUS_ERROR = 'ERROR';

    public function __construct(
        public readonly string $status,
        public readonly ?string $value = null,
        public readonly ?string $rawText = null,
        public readonly ?float $confidence = null,
        public readonly ?bool $formatValid = null,
        public readonly ?string $engineUsed = null,
        public readonly ?string $errorReason = null,
        public readonly ?string $matchMethod = null,
        public readonly ?int $processingTimeMs = null,
        public readonly array $raw = [],
    ) {}

    public static function fromResponse(array $data): self
    {
        $status = in_array($data['status'] ?? null, [self::STATUS_OK, self::STATUS_LOW_CONFIDENCE, self::STATUS_ERROR], true)
            ? $data['status']
            : self::STATUS_ERROR;

        $formatValid = $data['format_valid'] ?? null;

        return new self(
            status: $status,
            value: $data['extracted_date_code'] ?? ($formatValid === true ? ($data['raw_ocr_text'] ?? null) : null),
            rawText: $data['raw_ocr_text'] ?? null,
            confidence: isset($data['confidence']) ? (float) $data['confidence'] : null,
            formatValid: $formatValid,
            engineUsed: $data['engine_used'] ?? null,
            errorReason: $data['error_reason'] ?? ($status === self::STATUS_ERROR && ! isset($data['status']) ? 'UNEXPECTED_RESPONSE' : null),
            matchMethod: $data['match_method'] ?? null,
            processingTimeMs: isset($data['processing_time_ms']) ? (int) round($data['processing_time_ms']) : null,
            raw: $data,
        );
    }

    public static function error(string $reason): self
    {
        return new self(status: self::STATUS_ERROR, errorReason: $reason);
    }
}
