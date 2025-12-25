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
        Schema::create('tracks', function (Blueprint $table) {
            $table->char('id', 10)->primary();
            $table->char('songId', 10);
            $table->foreign('songId')
                ->references('id')
                ->on('Songs')
                ->cascadeOnDelete()
                ->cascadeOnUpdate();
            $table->char('playlistId', 10);
            $table->foreign('playlistId')
                ->references('id')
                ->on('Playlists')
                ->cascadeOnDelete()
                ->cascadeOnUpdate();
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('tracks');
    }
};
