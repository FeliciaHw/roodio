<?php
namespace App\Models;

use DB;
use Illuminate\Database\Eloquent\Model;

class Mood extends Model
{
    protected $fillable  = ['type'];
    public $incrementing = false;
    protected $keyType   = 'string';

    protected static function booted()
    {
        static::creating(function ($model) {

            $lastId = DB::table('moods')
                ->orderBy('id', 'desc')
                ->value('id');

            if (! $lastId) {
                $model->id = 'MD-0000001';
            } else {
                $number = (int) substr($lastId, 3);
                $number++;

                $model->id = 'MD-' . str_pad($number, 7, '0', STR_PAD_LEFT);
            }
        });
    }
}
