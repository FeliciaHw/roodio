<?php
namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Support\Facades\DB;

class Songs extends Model
{
    protected $fillable = [
        'userId',
        'moodId',
        'title',
        'lyrics',
        'artist',
        'genre',
        'duration',
        'publisher',
        'datePublished',
        'songPath',
    ];

    public $incrementing = false;
    protected $keyType   = 'string';

    public function mood(): BelongsTo
    {
        return $this->belongsTo(Mood::class);
    }

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    protected static function booted()
    {
        static::creating(function ($model) {

            $lastId = DB::table('songs')
                ->orderBy('id', 'desc')
                ->value('id');

            if (! $lastId) {
                $model->id = 'SG-0000001';
            } else {
                $number = (int) substr($lastId, 3);
                $number++;

                $model->id = 'SG-' . str_pad($number, 7, '0', STR_PAD_LEFT);
            }
        });
    }
}
