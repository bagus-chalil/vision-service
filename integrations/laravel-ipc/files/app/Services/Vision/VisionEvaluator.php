<?php

namespace App\Services\Vision;

use App\Models\IpcBatch;
use InvalidArgumentException;

/**
 * The PASS / FAIL / REVIEW decision for one Vision Service result. Lives in Laravel, never
 * in the Vision Service (settled IPC architecture). KPI is near-zero false PASS, so:
 *  - only an OK-confidence, format-valid read that matches the expected value is PASS;
 *  - ERROR, format mismatch, LOW_CONFIDENCE, missing reference data or a field_type with no
 *    rule in config('vision.rules') are always REVIEW — never forced into PASS or FAIL;
 *  - FAIL only when a confident, format-valid read disagrees with the expected value.
 * expectedValue / computedValue are still filled in on REVIEW whenever they can be derived,
 * so QC sees what the system expected while reviewing.
 */
class VisionEvaluator
{
    public function __construct(private readonly ExpDateCalculator $calculator) {}

    public function evaluate(IpcBatch $batch, string $fieldType, VisionResult $result): VisionEvaluation
    {
        if ($result->status === VisionResult::STATUS_ERROR) {
            return new VisionEvaluation(VisionEvaluation::REVIEW, 'Vision Service gagal memproses foto ('.($result->errorReason ?? 'unknown').') - foto ulang.');
        }

        if ($result->formatValid !== true || $result->value === null) {
            return new VisionEvaluation(VisionEvaluation::REVIEW, 'Kode tidak terbaca / format tidak sesuai ('.($result->errorReason ?? 'FORMAT_MISMATCH').') - foto ulang atau cek manual.');
        }

        $rule = config("vision.rules.{$fieldType}");

        [$expected, $computed, $problem] = match ($rule) {
            'batch_exp_date' => $this->expectedFromBatchExp($batch),
            'mfd_plus_shelf_life' => $this->expectedFromMfd($batch, $result->value),
            default => [null, null, "Belum ada aturan pembanding untuk field_type '{$fieldType}'."],
        };

        if ($problem !== null) {
            return new VisionEvaluation(VisionEvaluation::REVIEW, $problem, $expected, $computed);
        }

        $actual = $computed ?? $result->value;

        if ($result->status !== VisionResult::STATUS_OK) {
            return new VisionEvaluation(VisionEvaluation::REVIEW, 'Confidence OCR rendah - cek manual.', $expected, $computed);
        }

        return $actual === $expected
            ? new VisionEvaluation(VisionEvaluation::PASS, 'Sesuai dengan EXP batch.', $expected, $computed)
            : new VisionEvaluation(VisionEvaluation::FAIL, "Tidak sesuai: terbaca {$actual}, seharusnya {$expected}.", $expected, $computed);
    }

    /** @return array{0: ?string, 1: ?string, 2: ?string} expected, computed, problem */
    private function expectedFromBatchExp(IpcBatch $batch): array
    {
        if (! $batch->exp_date) {
            return [null, null, 'EXP date batch belum diisi (Startup Check) - tidak ada pembanding.'];
        }

        return [$batch->exp_date->format('dmy'), null, null];
    }

    /** @return array{0: ?string, 1: ?string, 2: ?string} expected, computed, problem */
    private function expectedFromMfd(IpcBatch $batch, string $mfd): array
    {
        $expected = $batch->exp_date?->format('dmy');
        $months = $batch->masterProduct?->shelf_life_months;

        if (! $months) {
            return [$expected, null, 'Shelf-life SKU belum diisi di master produk - EXP tidak bisa dihitung dari MFD.'];
        }

        try {
            $computed = $this->calculator->fromMfd($mfd, (int) $months);
        } catch (InvalidArgumentException $e) {
            return [$expected, null, $e->getMessage()];
        }

        if ($expected === null) {
            return [null, $computed, 'EXP date batch belum diisi (Startup Check) - tidak ada pembanding.'];
        }

        return [$expected, $computed, null];
    }
}
