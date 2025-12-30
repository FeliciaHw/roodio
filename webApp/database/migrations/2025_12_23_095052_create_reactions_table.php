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
        Schema::create('reactions', function (Blueprint $table) {
            $table->char('id', 10)->primary();
            $table->char('userId', 10);
            $table->foreign('userId')
                ->references('id')
                ->on('users')
                ->cascadeOnDelete()
                ->cascadeOnUpdate();
            $table->char('threadId', 10);
            $table->foreign('threadId')
                ->references('id')
                ->on('Threads')
                ->cascadeOnDelete()
                ->cascadeOnUpdate();
            $table->boolean('isReacted')->default(false);
            $table->unique(['userId', 'threadId']);
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('reactions');
    }
};
