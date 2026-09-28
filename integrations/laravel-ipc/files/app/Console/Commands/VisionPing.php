<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use Illuminate\Http\Client\ConnectionException;
use Illuminate\Support\Facades\Http;

/**
 * `php artisan vision:ping` — run on the Laravel server after deploy to prove it can reach
 * the Vision Service (firewall allowlist on the VPS only admits 10.10.162.0/24 and
 * 100.100.160.0/24 on port 8000).
 */
class VisionPing extends Command
{
    protected $signature = 'vision:ping';

    protected $description = 'Cek koneksi ke Vision Service (health + daftar field_type)';

    public function handle(): int
    {
        $base = rtrim(config('vision.base_url'), '/');
        $this->line("Vision Service: {$base}");

        try {
            $health = Http::timeout(10)->connectTimeout(config('vision.connect_timeout'))->get("{$base}/api/health");
            $types = Http::timeout(10)->connectTimeout(config('vision.connect_timeout'))->get("{$base}/api/field-types");
        } catch (ConnectionException $e) {
            $this->error('Tidak bisa terhubung: '.$e->getMessage());

            return self::FAILURE;
        }

        if (! $health->successful() || $health->json('status') !== 'ok') {
            $this->error("Health check gagal (HTTP {$health->status()}).");

            return self::FAILURE;
        }

        $this->info('Health: ok');

        $available = collect($types->json() ?? [])->pluck('key');
        foreach (config('vision.field_types') as $fieldType) {
            $available->contains($fieldType)
                ? $this->info("  [ada]   {$fieldType}")
                : $this->error("  [TIDAK ADA di Vision Service] {$fieldType}");
        }

        return collect(config('vision.field_types'))->diff($available)->isEmpty() ? self::SUCCESS : self::FAILURE;
    }
}
