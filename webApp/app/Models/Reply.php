<?php
namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Support\Facades\DB;

class Reply extends Model
{
    protected $fillable  = ['content', 'threadId', 'userId'];
    public $incrementing = false;
    protected $keyType   = 'string';

    protected static function booted()
    {
        static::creating(function ($model) {

            $lastId = DB::table('replies')
                ->orderBy('id', 'desc')
                ->value('id');

            if (! $lastId) {
                $model->id = 'RP-0000001';
            } else {
                $number = (int) substr($lastId, 3);
                $number++;

                $model->id = 'RP-' . str_pad($number, 7, '0', STR_PAD_LEFT);
            }
        });
    }
}
