<?php
namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Models\Songs;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Storage;

class SongController extends Controller
{
    /**
     * Display a listing of the resource.
     */
    public function index()
    {
        return view('admin.songs.index');
    }

    /**
     * Show the form for creating a new resource.
     */
    public function create()
    {
        return view('admin.songs.create');
    }

    /**
     * Store a newly created resource in storage.
     */
    public function store(Request $request)
    {
        $validated = $request->validate([
            'title'         => 'required|max:255',
            'lyrics'        => 'required',
            'artist'        => 'required|max:255',
            'genre'         => 'required|max:255',
            'publisher'     => 'required|max:255',
            'datePublished' => 'required|date',
            'song'          => 'required|file|mimes:mp3',
        ]);

        $song = $request->file('song');
        $path = Storage::disk('azure')->put(
            'songs',
            $song
        );

        $datas             = $validated;
        $datas['duration'] = 120;
        $datas['songPath'] = $path;
        $datas['userId']   = Auth::id();
        $datas['moodId']   = 'MD-0000001';
        unset($datas['song']);

        Songs::create($datas);
        return redirect()->route('admin.songs.index')->with('succes', 'Successfully added song');
    }

    /**
     * Display the specified resource.
     */
    public function show(string $id)
    {
        //
    }

    /**
     * Show the form for editing the specified resource.
     */
    public function edit(string $id)
    {
        //
    }

    /**
     * Update the specified resource in storage.
     */
    public function update(Request $request, string $id)
    {
        //
    }

    /**
     * Remove the specified resource from storage.
     */
    public function destroy(string $id)
    {
        //
    }
}
