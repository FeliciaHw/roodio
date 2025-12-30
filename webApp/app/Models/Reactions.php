<?php
namespace App\Models;

use DB;
use Illuminate\Database\Eloquent\Model;

class Reactions extends Model
{
    protected $fillable  = ['userId', 'threadId'];
    public $incrementing = false;
    protected $keyType   = 'string';

    protected static function booted()
    {
        static::creating(function ($model) {

            $lastId = DB::table('reactions')
                ->orderBy('id', 'desc')
                ->value('id');

            if (! $lastId) {
                $model->id = 'RC-0000001';
            } else {
                $number = (int) substr($lastId, 3);
                $number++;

                $model->id = 'RC-' . str_pad($number, 7, '0', STR_PAD_LEFT);
            }
        });
    }
}
