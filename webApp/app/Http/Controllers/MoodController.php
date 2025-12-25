<?php
namespace App\Http\Controllers;

use App\Models\Mood;
use Illuminate\Http\Request;
use Illuminate\Validation\Rule;

class MoodController extends Controller
{
    /**
     * Display a listing of the resource.
     */
    public function index()
    {
        $moods = Mood::orderByDesc('created_at')->paginate(5);
        return view('moods.index', compact('moods'));
    }

    /**
     * Show the form for creating a new resource.
     */
    public function create()
    {
        return view('moods.create');
    }

    /**
     * Store a newly created resource in storage.
     */
    public function store(Request $request)
    {
        $request->validate([
            'type' => 'required|string|min:2|max:255|unique:moods,type',
        ]);

        Mood::create($request->all());
        return redirect()->route('moods.index')->with('success', 'Mood added successfully');
    }

    /**
     * Show the form for editing the specified resource.
     */
    public function edit(Mood $mood)
    {
        return view('moods.edit', compact('mood'));
    }

    /**
     * Update the specified resource in storage.
     */
    public function update(Request $request, Mood $mood)
    {
        $request->validate([
            'type' => [
                'required',
                Rule::unique('moods', 'type')->ignore($mood->id),
            ],
        ]);

        $mood->update($request->all());
        return redirect()->route('moods.index')->with('success', 'Mood updated successfully');
    }

    /**
     * Remove the specified resource from storage.
     */
    public function destroy(Mood $mood)
    {
        $mood->delete();
        return redirect()->route('moods.index')->with('success', 'Mood deleted successfully');
    }
}
