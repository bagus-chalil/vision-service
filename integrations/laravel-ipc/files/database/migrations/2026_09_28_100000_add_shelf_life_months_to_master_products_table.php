<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

// Shelf-life is per SKU (fg_code), so it lives on master_products. Nullable: a product with
// no shelf-life filled in simply can't have its EXP derived from MFD (VisionEvaluator -> REVIEW).
return new class extends Migration
{
    public function up(): void
    {
        Schema::table('master_products', function (Blueprint $table) {
            $table->unsignedSmallInteger('shelf_life_months')->nullable()->after('product_name');
        });
    }

    public function down(): void
    {
        Schema::table('master_products', function (Blueprint $table) {
            $table->dropColumn('shelf_life_months');
        });
    }
};
