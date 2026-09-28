<?php

namespace Tests\Unit;

use App\Services\Vision\ExpDateCalculator;
use InvalidArgumentException;
use PHPUnit\Framework\TestCase;

class ExpDateCalculatorTest extends TestCase
{
    public function test_adds_shelf_life_months_to_mfd(): void
    {
        // Real sample 2026-09-28: Pond's UV Protect, MFD 140624 printed on the body, EXP
        // 140627 embossed on the crimp -> 36 months.
        $this->assertSame('140627', (new ExpDateCalculator)->fromMfd('140624', 36));
        $this->assertSame('150125', (new ExpDateCalculator)->fromMfd('151224', 1));
    }

    public function test_flags_a_day_missing_from_the_target_month_instead_of_rolling_over(): void
    {
        $this->expectException(InvalidArgumentException::class);

        (new ExpDateCalculator)->fromMfd('310124', 1);
    }

    public function test_leap_day_only_valid_when_target_year_is_a_leap_year(): void
    {
        $this->assertSame('290228', (new ExpDateCalculator)->fromMfd('290224', 48));

        $this->expectException(InvalidArgumentException::class);
        (new ExpDateCalculator)->fromMfd('290224', 12);
    }

    public function test_rejects_an_impossible_mfd_date(): void
    {
        $this->expectException(InvalidArgumentException::class);

        (new ExpDateCalculator)->fromMfd('310224', 36);
    }
}
