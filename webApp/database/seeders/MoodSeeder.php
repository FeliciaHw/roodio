<?php
namespace Database\Seeders;

use App\Models\Mood;
use Illuminate\Database\Seeder;

class MoodSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        $moods = [
            ['id' => 'MD-0000001', 'type' => 'happy'],
            ['id' => 'MD-0000002', 'type' => 'sad'],
            ['id' => 'MD-0000003', 'type' => 'relax'],
            ['id' => 'MD-0000004', 'type' => 'angry'],
        ];

        Mood::insert($moods);
    }
}
