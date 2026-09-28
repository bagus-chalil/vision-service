<?php

namespace App\Services\Vision;

class VisionEvaluation
{
    public const PASS = 'PASS';

    public const FAIL = 'FAIL';

    public const REVIEW = 'REVIEW';

    public function __construct(
        public readonly string $decision,
        public readonly string $reason,
        public readonly ?string $expectedValue = null,
        public readonly ?string $computedValue = null,
    ) {}
}
