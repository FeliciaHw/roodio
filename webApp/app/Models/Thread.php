<?php
namespace App\Models;

use DB;
use Illuminate\Database\Eloquent\Model;

class Thread extends Model
{
    protected $fillable  = ['userId', 'title', 'content', 'isReplyable', 'datePost'];
    public $incrementing = false;
    protected $keyType   = 'string';

    protected static function booted()
    {
        static::creating(function ($model) {

            $lastId = DB::table('threads')
                ->orderBy('id', 'desc')
                ->value('id');

            if (! $lastId) {
                $model->id = 'TH-0000001';
            } else {
                $number = (int) substr($lastId, 3);
                $number++;

                $model->id = 'TH-' . str_pad($number, 7, '0', STR_PAD_LEFT);
            }
        });
    }
}
