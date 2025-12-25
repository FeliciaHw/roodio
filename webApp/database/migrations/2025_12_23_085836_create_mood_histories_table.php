<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::create('mood_histories', function (Blueprint $table) {
            $table->char('id', 10)->primary();
            $table->char('userId', 10);
            $table->foreign('userId')
                ->references('id')
                ->on('users')
                ->cascadeOnDelete()
                ->cascadeOnUpdate();
            $table->char('moodId', 10);
            $table->foreign('moodId')
                ->references('id')
                ->on('Moods')
                ->noActionOnDelete()
                ->noActionOnUpdate();
            $table->integer('count');
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('mood_histories');
    }
};
